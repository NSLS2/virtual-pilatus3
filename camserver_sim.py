#!/usr/bin/env python3
"""
Pilatus 3 Camserver Protocol Simulator (Extended)

Implements (a best-effort simulation of) many commands listed in the
PILATUS3 R/S/X User Manual v1.2.0 (Table 13.1 excerpt provided by user).

This simulator:
- Speaks a simplified line-oriented TCP protocol on a configurable port.
- Returns numeric "return codes" plus textual messages meant to resemble
  the real camserver (but is NOT exact / official).
- Simulates exposure sequences (internal, external trigger variants).
- Provides asynchronous acknowledgements for image completion (code 7)
  and optional periodic acknowledgements via SetAckInt.
- Supports a large subset of the configuration / query commands:
  Exposure, ExtTrigger, ExtMtrigger, ExtEnable, ExpTime, ExpPeriod, ImgPath,
  NImages, Delay, NExpFrame, MXsettings, SetCu/Mo/Cr/Fe/Ag/Ga, SetThreshold,
  SetEnergy, K, camcmd (K), LdBadPixMap, LdFlatField, RateCorrLUTDir,
  ReadoutTime, SetRetriggerMode, GapFill, THread, SetAckInt, ResetCam,
  DebTime, HeaderString, Df, ExpEnd, CamSetup, Telemetry, ResetModulePower,
  Version, ShowPID, Exit/Quit.

Caveats / Simplifications:
- External trigger modes (ExtTrigger / ExtMtrigger / ExtEnable) are simulated:
  real hardware waits indefinitely; here we auto-fire triggers unless you set
  --manual-triggers and use TRIGGER command (non-standard helper) to inject them.
- Timing is approximate; readout time is configurable and static.
- Threshold / energy trimming commands create pseudo "trim file" paths only.
- Temperature/humidity (THread) values are random plausible numbers.
- Rate correction / flat field / bad pixel maps are not applied, only stored.
- Error handling is minimal; command parsing tolerant but not identical to real device.
- Not all manual commands are implemented (only those listed in the provided excerpt).

Return Code Conventions (approximate):
  2  CamSetup report
  5  Disk free (Df)
  6  ExpEnd
  7  Exposure (or series) completion ack / per-image ack
  10 ImgPath path echo
  13 Interrupt acknowledgement (K / camcmd K)
  15 Generic OK / setting change / command acceptance
  16 PID
  18 Telemetry
  24 Version
  215 THread sensor read
  > Any error: 999 (simulator-specific, not in manual) plus message starting with ERR

Protocol:
- Client connects, receives banner lines (prefixed with '# ' for comments).
- Client sends command lines (ASCII). Arguments separated by spaces.
- Responses are single lines starting with a numeric return code, a space,
  then textual message (unless noted).
- Asynchronous acknowledgements (image completion) appear as their own lines
  with code 7 and message "full path name of image".
- Lines starting with "# " are informational (not part of official camserver format).

Helper (non-standard) commands added for simulation:
  HELP        - list implemented commands.
  TRIGGER     - inject an external trigger (if waiting and in a manual trigger mode).
  STATE       - show internal simulator state (debug).
  ABORT       - alias for K.

Usage:
  python pilatus_camserver_full_sim.py --port 41234

Examples (netcat):
  $ nc localhost 41234
  ExpTime 0.2
  15 Exposure time set to: 0.200 sec
  NImages 3
  15 N images set to: 3
  Exposure test.cbf
  15 starting 0.200 second background: 2025-08-28 02:54:00
  (image acks)
  7 /ramdisk/test_000003.cbf    <-- final ack, last image path

Author: Jakub Wlodek
"""

import argparse
import asyncio
import dataclasses
import logging
import os
import random
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, Awaitable, List

# Used for generating random image data
import numpy as np
import tiffile as tf

__version__ = "1.0.0"

# ------------------ Constants ------------------

MAX_IMAGES = 999_999
DEFAULT_IMGPATH = Path("/ramdisk")
DEFAULT_EXPOSURE_TIME = 1.0
DEFAULT_EXPOSURE_PERIOD = 1.05  # Must be >= exposure time + readout
DEFAULT_READOUT_TIME = 0.005  # seconds (simulated)
ACK_CODE_OK_GENERIC = 15
ACK_CODE_IMAGE_DONE = 7
ACK_CODE_INTERRUPT = 13
ACK_CODE_DISKFREE = 5
ACK_CODE_EXPEND = 6
ACK_CODE_CAMSETUP = 2
ACK_CODE_THREAD = 215
ACK_CODE_PID = 16
ACK_CODE_TELEMETRY = 18
ACK_CODE_VERSION = 24
ACK_CODE_IMGPATH = 10
ERROR_CODE = 999  # Simulator-specific (not in manual)

ELEMENT_TRIM_SETTINGS = {
    "Cu": 8048,
    "Mo": 17479,
    "Cr": 5414,
    "Fe": 6405,
    "Ag": 22163,
    "Ga": 9658,
}

# ------------------ Data Models ------------------


@dataclasses.dataclass
class ThresholdSettings:
    incident_energy_eV: Optional[int] = None
    threshold_eV: Optional[int] = None
    gain: Optional[str] = None
    trim_file: Optional[str] = None
    vcmp: Optional[float] = None  # Simulated comparator voltage


@dataclasses.dataclass
class MXSettings:
    values: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DetectorParams:
    exposure_time: float = DEFAULT_EXPOSURE_TIME
    exposure_period: float = DEFAULT_EXPOSURE_PERIOD
    n_images: int = 1
    delay: float = 0.0
    n_exp_frame: int = 1
    ack_interval: int = 0
    retrigger_mode: bool = True
    gap_fill: int = 0
    debounce_time: float = 0.0
    header_string: Optional[str] = None
    ratecorr_dir: Optional[str] = None
    badpix_map: Optional[str] = None
    flatfield_file: Optional[str] = None
    threshold: ThresholdSettings = dataclasses.field(default_factory=ThresholdSettings)
    mx: MXSettings = dataclasses.field(default_factory=MXSettings)


@dataclasses.dataclass
class AcquisitionState:
    running: bool = False
    external_mode: Optional[str] = None  # 'ExtTrigger', 'ExtMtrigger', 'ExtEnable'
    armed: bool = False
    aborted: bool = False
    current_index: int = 0
    last_image_path: Optional[str] = None
    start_time: Optional[float] = None
    task: Optional[asyncio.Task] = None
    waiting_for_trigger: bool = False
    reset_in_progress: bool = False
    module_power_reset: bool = False


# ------------------ Simulator Core ------------------


class CamserverSimulator:
    def __init__(
        self,
        xsize: int,
        ysize: int,
        host: str,
        port: int,
        terminator: str,
        manual_triggers: bool,
        readout_time: float,
        loop: asyncio.AbstractEventLoop,
    ):
        self.xsize = xsize
        self.ysize = ysize
        self.host = host
        self.port = port
        # Interpret common escape sequences in terminator argument so users can pass "\\n", "\\r\n", "\\t", etc.
        try:
            processed = bytes(terminator, "utf-8").decode("unicode_escape")
        except Exception:
            processed = terminator  # fallback (use raw string)
        if processed == "":  # never allow empty terminator (would hang)
            processed = "\n"
        self.terminator = processed
        self._terminator_bytes = processed.encode()
        self.manual_triggers = manual_triggers
        self.readout_time = readout_time
        self.loop = loop
        self.params = DetectorParams()
        self.state = AcquisitionState()
        self.img_path: Path = DEFAULT_IMGPATH
        self.clients: List[asyncio.StreamWriter] = []
        self.logger = logging.getLogger("PilatusFullSim")
        self.lock = asyncio.Lock()
        self.pid = os.getpid()
        self.version_string = f"TVX/Camserver Sim {__version__}"
        self._register_time = time.time()

        # Pre-build command dispatcher
        self.commands: Dict[
            str, Callable[[str, asyncio.StreamWriter], Awaitable[None]]
        ] = {
            "Exposure": self.cmd_exposure,
            "ExtTrigger": self.cmd_ext_trigger,
            "ExtMtrigger": self.cmd_ext_mtrigger,
            "ExtEnable": self.cmd_ext_enable,
            "ExpTime": self.cmd_exp_time,
            "ExpPeriod": self.cmd_exp_period,
            "ImgPath": self.cmd_img_path,
            "NImages": self.cmd_n_images,
            "Delay": self.cmd_delay,
            "NExpFrame": self.cmd_n_exp_frame,
            "MXsettings": self.cmd_mxsettings,
            "SetCu": self.cmd_set_element("Cu"),
            "SetMo": self.cmd_set_element("Mo"),
            "SetCr": self.cmd_set_element("Cr"),
            "SetFe": self.cmd_set_element("Fe"),
            "SetAg": self.cmd_set_element("Ag"),
            "SetGa": self.cmd_set_element("Ga"),
            "SetThreshold": self.cmd_set_threshold,
            "SetEnergy": self.cmd_set_energy,
            "K": self.cmd_k,
            "camcmd": self.cmd_camcmd,  # expects argument 'K'
            "LdBadPixMap": self.cmd_ld_badpix,
            "LdFlatField": self.cmd_ld_flatfield,
            "RateCorrLUTDir": self.cmd_ratecorr_lutdir,
            "ReadoutTime": self.cmd_readout_time,
            "SetRetriggerMode": self.cmd_set_retrigger,
            "GapFill": self.cmd_gap_fill,
            "THread": self.cmd_thread,
            "SetAckInt": self.cmd_ack_int,
            "ResetCam": self.cmd_reset_cam,
            "DebTime": self.cmd_deb_time,
            "HeaderString": self.cmd_header_string,
            "Exit": self.cmd_exit,
            "Quit": self.cmd_exit,
            "Df": self.cmd_df,
            "ExpEnd": self.cmd_exp_end,
            "CamSetup": self.cmd_cam_setup,
            "Telemetry": self.cmd_telemetry,
            "ResetModulePower": self.cmd_reset_module_power,
            "Version": self.cmd_version,
            "ShowPID": self.cmd_show_pid,
            # Helper / simulator additions
            "HELP": self.cmd_help,
            "STATE": self.cmd_state,
            "TRIGGER": self.cmd_trigger,
            "ABORT": self.cmd_k,
        }

    # ---------- Utility Output Methods ----------

    async def send_line(self, writer: asyncio.StreamWriter, code: int, text: str):
        line = f"{code} {text}{self.terminator}"
        writer.write(line.encode())
        await writer.drain()

    async def broadcast_ack(self, code: int, text: str):
        dead = []
        for w in self.clients:
            try:
                w.write(f"{code} {text}{self.terminator}".encode())
                await w.drain()
            except Exception:
                dead.append(w)
        for w in dead:
            self.clients.remove(w)

    async def info(self, writer: asyncio.StreamWriter, msg: str):
        writer.write(f"# {msg}{self.terminator}".encode())
        await writer.drain()

    def now_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- Filename Generation ----------

    def build_image_path(self, base_name: str, index: int, total: int) -> Path:
        """
        Insert image number before extension if total > 1.
        Zero pad to 6 digits (as is typical).
        """
        p = Path(base_name)
        directory = self.img_path
        if p.is_absolute():
            directory = p.parent
            stem = p.stem
            ext = p.suffix
        else:
            stem = p.stem
            ext = p.suffix
        if not ext:
            ext = ".raw"
        if total > 1:
            num = f"{index + 1:06d}"
            filename = f"{stem}_{num}{ext}"
        else:
            filename = f"{stem}{ext}"
        return directory / filename

    # ---------- Exposure Core ----------

    async def start_exposure_series(
        self, mode: str, writer: asyncio.StreamWriter, base_name: str
    ):
        async with self.lock:
            if self.state.running or self.state.armed:
                await self.send_line(writer, ERROR_CODE, "ERR Busy")
                return
            # Validate period vs exposure time + readout
            if (
                self.params.exposure_period
                < self.params.exposure_time + self.readout_time
            ):
                await self.send_line(
                    writer, ERROR_CODE, "ERR ExpPeriod too short for exposure + readout"
                )
                return
            self.state.running = mode == "Exposure"
            self.state.external_mode = None if mode == "Exposure" else mode
            self.state.armed = mode != "Exposure"
            self.state.aborted = False
            self.state.current_index = 0
            self.state.last_image_path = None
            self.state.start_time = time.time()
            self._current_base_name = base_name
            self.logger.info(
                "Starting mode=%s n_images=%d base=%s",
                mode,
                self.params.n_images,
                base_name,
            )

        # Start messages
        if mode == "Exposure":
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"starting {self.params.exposure_time:.3f} second background: {self.now_str()}",
            )
            # Launch internal task
            self.state.task = self.loop.create_task(self._run_internal_series(writer))
        elif mode == "ExtTrigger":
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"starting externally triggered exposure(s): {self.now_str()}",
            )
            # Wait for external triggers (one per series) -> entire series started by one trigger
            self.state.task = self.loop.create_task(
                self._run_ext_trigger_series(writer)
            )
        elif mode == "ExtMtrigger":
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"starting externally multi-triggered exposure(s): {self.now_str()}",
            )
            self.state.task = self.loop.create_task(
                self._run_ext_mtrigger_series(writer)
            )
        elif mode == "ExtEnable":
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"starting externally enabled exposure(s): {self.now_str()}",
            )
            self.state.task = self.loop.create_task(self._run_ext_enable_series(writer))

    async def _respect_ack_interval(self, img_path: Path, img_index: int):
        """
        Emit per-image ack if ack_interval condition satisfied or always for final image.
        """
        total = self.params.n_images
        ack_every = self.params.ack_interval
        if ack_every == 0:
            # Only final image
            if img_index == total - 1:
                await self.broadcast_ack(ACK_CODE_IMAGE_DONE, str(img_path))
        else:
            if (img_index + 1) % ack_every == 0 or img_index == total - 1:
                await self.broadcast_ack(ACK_CODE_IMAGE_DONE, str(img_path))

    async def _simulate_single_exposure(self, img_index: int, total: int) -> Path:
        # Delay for external modes (first exposure)
        if img_index == 0 and self.params.delay > 0:
            await asyncio.sleep(self.params.delay)
        # Exposure (simulate)
        await asyncio.sleep(self.params.exposure_time)
        # Readout/writing
        await asyncio.sleep(self.readout_time)
        img_path = self.build_image_path(self._current_base_name, img_index, total)
        # "Write" file (touch)
        try:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if os.path.splitext(img_path)[1] == ".tif":
                # Create a dummy tif file with random data.
                # pilatus produces 32 bit signed integer data, but data is generally in a 16 bit range.
                data = np.random.randint(
                    0, 65536, size=(self.ysize, self.xsize), dtype=np.int32
                )
                tf.imwrite(img_path, data)
            else:
                # For anything but tif files, just create an empty one.
                img_path.touch()
        except Exception as e:
            self.logger.error(f"Failed to write image file {img_path}: {e}")
        return img_path

    async def _run_internal_series(self, writer: asyncio.StreamWriter):
        total = self.params.n_images
        for i in range(total):
            async with self.lock:
                if self.state.aborted:
                    break
            img_path = await self._simulate_single_exposure(i, total)
            async with self.lock:
                self.state.current_index = i
                self.state.last_image_path = str(img_path)
            await self._respect_ack_interval(img_path, i)
            # Inter-frame wait: exposure_period - exposure_time - readout
            remain = (
                self.params.exposure_period
                - self.params.exposure_time
                - self.readout_time
            )
            if remain > 0 and i < total - 1:
                await asyncio.sleep(remain)
            async with self.lock:
                if self.state.aborted:
                    break
        await self._finalize_series(writer)

    async def _run_ext_trigger_series(self, writer: asyncio.StreamWriter):
        """
        In ExtTrigger: A single external trigger starts the entire series (internal timing).
        We either wait for manual TRIGGER or auto-fire after a short wait if manual_triggers is False.
        """
        if not self.manual_triggers:
            await asyncio.sleep(self.params.delay)
            await self._inject_trigger()
        # Otherwise we rely on TRIGGER command to call _inject_trigger.
        # Acquisition completion handled inside _inject_trigger-driven internal series.

    async def _run_ext_mtrigger_series(self, writer: asyncio.StreamWriter):
        """
        Each exposure started by an external trigger.
        """
        remaining = self.params.n_images
        if not self.manual_triggers:
            # Auto-fire triggers according to exposure_period
            for _ in range(remaining):
                async with self.lock:
                    if self.state.aborted:
                        break
                await self._inject_trigger(single=True)
                await asyncio.sleep(self.params.exposure_period)
        # Manual triggers rely on TRIGGER.

        if self.state.running:
            # Already completed by triggers
            return

        # If we've not yet transitioned to running but all images triggered, finalize.
        async with self.lock:
            if (
                self.state.current_index + 1 >= self.params.n_images
                and self.state.last_image_path
            ):
                await self._finalize_series(writer)

    async def _run_ext_enable_series(self, writer: asyncio.StreamWriter):
        """
        ExtEnable: exposures bracketed by enable high/low.
        We approximate by turning each trigger into an exposure of the configured ExpTime.
        """
        if not self.manual_triggers:
            for i in range(self.params.n_images):
                async with self.lock:
                    if self.state.aborted:
                        break
                await self._inject_trigger(single=True)
                await asyncio.sleep(self.params.exposure_period)
        # manual path relies on TRIGGER
        async with self.lock:
            if (
                self.state.current_index + 1 >= self.params.n_images
                and self.state.last_image_path
            ):
                await self._finalize_series(writer)

    async def _inject_trigger(self, single: bool = False):
        """
        Simulate arrival of an external trigger.
        For ExtTrigger: first trigger starts the entire internal sequence
        For ExtMtrigger / ExtEnable: each trigger fires one exposure.
        """
        async with self.lock:
            if self.state.aborted or (
                not self.state.armed and not self.state.external_mode
            ):
                return
            mode = self.state.external_mode

        if mode == "ExtTrigger":
            # Start internal series
            async with self.lock:
                self.state.running = True
                self.state.armed = False
            # Run same as internal
            dummy_writer = None
            await self._run_internal_series(
                dummy_writer
            )  # writer not needed during steps
        elif mode in ("ExtMtrigger", "ExtEnable"):
            async with self.lock:
                total = self.params.n_images
                img_index = self.state.current_index
                if img_index >= total:
                    return
                # Mark as 'running' after first trigger
                self.state.running = True
                self.state.armed = True  # remain armed until sequence done
            img_path = await self._simulate_single_exposure(img_index, total)
            async with self.lock:
                self.state.last_image_path = str(img_path)
                self.state.current_index += 1
                final = self.state.current_index >= total
            await self._respect_ack_interval(img_path, img_index)
            if final:
                async with self.lock:
                    self.state.running = False
                    self.state.armed = False
                await self.broadcast_ack(ACK_CODE_IMAGE_DONE, str(img_path))
        # After exposures if finished finalize
        async with self.lock:
            if (
                not self.state.running
                and not self.state.armed
                and self.state.last_image_path
            ):
                # Use last writer (none) -> broadcast final ack once only if not already final
                pass

    async def _finalize_series(self, writer: Optional[asyncio.StreamWriter]):
        async with self.lock:
            last_path = self.state.last_image_path
            self.state.running = False
            self.state.armed = False
        if last_path:
            # Final ack already handled via ack interval; still ensure final ack present
            await self.broadcast_ack(ACK_CODE_IMAGE_DONE, last_path)

    async def abort_series(self):
        async with self.lock:
            if not (self.state.running or self.state.armed):
                return None
            self.state.aborted = True
            last = self.state.last_image_path
        return last

    # ---------- Command Implementations ----------

    async def cmd_exposure(self, args: str, writer: asyncio.StreamWriter):
        base = args.strip() or "image.cbf"
        await self.start_exposure_series("Exposure", writer, base)

    async def cmd_ext_trigger(self, args: str, writer: asyncio.StreamWriter):
        base = args.strip() or "exttrig.cbf"
        await self.start_exposure_series("ExtTrigger", writer, base)

    async def cmd_ext_mtrigger(self, args: str, writer: asyncio.StreamWriter):
        base = args.strip() or "extmtrig.cbf"
        await self.start_exposure_series("ExtMtrigger", writer, base)

    async def cmd_ext_enable(self, args: str, writer: asyncio.StreamWriter):
        base = args.strip() or "extenable.cbf"
        await self.start_exposure_series("ExtEnable", writer, base)

    async def cmd_exp_time(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"Exposure time set to: {self.params.exposure_time:.3f} sec",
            )
            return
        try:
            val = float(args)
            if val <= 0 or val > 60 * 60 * 24 * 60:  # 60 days
                raise ValueError
            self.params.exposure_time = val
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"Exposure time set to: {val:.3f} sec"
            )
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid exposure time")

    async def cmd_exp_period(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"Exposure period set to: {self.params.exposure_period:.3f} sec",
            )
            return
        try:
            val = float(args)
            if val <= 0 or val > 60 * 60 * 24 * 60:
                raise ValueError
            self.params.exposure_period = val
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"Exposure period set to: {val:.3f} sec"
            )
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid exposure period")

    async def cmd_img_path(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(writer, ACK_CODE_IMGPATH, str(self.img_path))
            return
        new_path = Path(args.strip())
        try:
            if not new_path.is_absolute():
                new_path = (self.img_path / new_path).resolve()
            new_path.mkdir(parents=True, exist_ok=True)
            self.img_path = new_path
            await self.send_line(writer, ACK_CODE_IMGPATH, str(self.img_path))
        except Exception as e:
            await self.send_line(writer, ERROR_CODE, f"ERR cannot set ImgPath: {e}")

    async def cmd_n_images(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"N images set to: {self.params.n_images}"
            )
            return
        try:
            val = int(args)
            if val <= 0 or val > MAX_IMAGES:
                raise ValueError
            self.params.n_images = val
            await self.send_line(writer, ACK_CODE_OK_GENERIC, f"N images set to: {val}")
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid number of images")

    async def cmd_delay(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"Delay time set to: {self.params.delay:.3f} sec",
            )
            return
        try:
            val = float(args)
            if val < 0 or val >= 64:
                raise ValueError
            self.params.delay = val
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"Delay time set to: {val:.3f} sec"
            )
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid delay time")

    async def cmd_n_exp_frame(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"Exposures per frame set to: {self.params.n_exp_frame}",
            )
            return
        try:
            val = int(args)
            if val < 1:
                raise ValueError
            self.params.n_exp_frame = val
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"Exposures per frame set to: {val}"
            )
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid NExpFrame")

    async def cmd_mxsettings(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            if not self.params.mx.values:
                await self.send_line(writer, ACK_CODE_OK_GENERIC, "None set")
            else:
                joined = " ".join(f"{k} {v}" for k, v in self.params.mx.values.items())
                await self.send_line(writer, ACK_CODE_OK_GENERIC, joined)
            return
        toks = args.split()
        if len(toks) % 2 != 0:
            await self.send_line(
                writer, ERROR_CODE, "ERR MXsettings requires key value pairs"
            )
            return
        for k, v in zip(toks[::2], toks[1::2]):
            self.params.mx.values[k] = v
        await self.send_line(writer, ACK_CODE_OK_GENERIC, "MX settings updated")

    def cmd_set_element(self, symbol: str):
        async def handler(args: str, writer: asyncio.StreamWriter):
            energy = ELEMENT_TRIM_SETTINGS[symbol]
            # Simulate new threshold ~ 0.9 * energy
            threshold = int(0.9 * energy)
            trim_file = f"/tmp/setthreshold_{symbol.lower()}.cmd"
            self.params.threshold.incident_energy_eV = energy
            self.params.threshold.threshold_eV = threshold
            self.params.threshold.trim_file = trim_file
            self.params.threshold.gain = "mid"
            self.params.threshold.vcmp = round(random.uniform(0.400, 0.900), 3)
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, "OK /tmp/setthreshold.cmd"
            )

        return handler

    async def cmd_set_threshold(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            t = self.params.threshold
            if t.threshold_eV is None:
                await self.send_line(
                    writer, ACK_CODE_OK_GENERIC, "Threshold has not been set"
                )
            else:
                msg = (
                    f"Settings: {t.gain or 'mid'} gain; threshold: {t.threshold_eV} eV; "
                    f"vcmp: {t.vcmp or 0:.3f} V Trim file: {t.trim_file}"
                )
                await self.send_line(writer, ACK_CODE_OK_GENERIC, msg)
            return
        toks = args.split()
        # Simple forms: SetThreshold 7400 OR SetThreshold energy 14200 7500
        if len(toks) == 1:
            try:
                thr = int(toks[0])
                if thr == 0:
                    self.params.threshold = ThresholdSettings()
                    await self.send_line(
                        writer, ACK_CODE_OK_GENERIC, "Threshold invalidated"
                    )
                    return
                self.params.threshold.threshold_eV = thr
                self.params.threshold.incident_energy_eV = thr * 2
                self.params.threshold.gain = "mid"
                self.params.threshold.vcmp = round(random.uniform(0.400, 0.900), 3)
                self.params.threshold.trim_file = f"/tmp/trim_thr{thr}.bin"
                await self.send_line(
                    writer,
                    ACK_CODE_OK_GENERIC,
                    f"Setting the threshold: {self.params.threshold.trim_file}",
                )
            except ValueError:
                await self.send_line(writer, ERROR_CODE, "ERR invalid threshold")
            return
        # Parse key-value style
        incident = None
        threshold = None
        gain = None
        i = 0
        while i < len(toks):
            key = toks[i]
            if key == "energy":
                i += 1
                if i >= len(toks):
                    await self.send_line(writer, ERROR_CODE, "ERR missing energy value")
                    return
                try:
                    incident = int(toks[i])
                except ValueError:
                    await self.send_line(writer, ERROR_CODE, "ERR invalid energy value")
                    return
            elif key.lower().endswith("g"):
                # gain threshold pair (legacy)
                gain = key
                i += 1
                if i >= len(toks):
                    await self.send_line(
                        writer, ERROR_CODE, "ERR missing threshold after gain"
                    )
                    return
                try:
                    threshold = int(toks[i])
                except ValueError:
                    await self.send_line(writer, ERROR_CODE, "ERR invalid threshold")
                    return
            else:
                # Assume threshold
                try:
                    threshold = int(key)
                except ValueError:
                    await self.send_line(writer, ERROR_CODE, f"ERR unknown token {key}")
                    return
            i += 1
        if threshold is None:
            await self.send_line(writer, ERROR_CODE, "ERR threshold not specified")
            return
        if incident is None:
            incident = 2 * threshold
        t = self.params.threshold
        t.threshold_eV = threshold
        t.incident_energy_eV = incident
        t.gain = gain or "mid"
        t.vcmp = round(random.uniform(0.400, 0.900), 3)
        t.trim_file = f"/tmp/trim_energy{incident}_thr{threshold}.bin"
        await self.send_line(
            writer, ACK_CODE_OK_GENERIC, f"Setting the threshold: {t.trim_file}"
        )

    async def cmd_set_energy(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            t = self.params.threshold
            if t.threshold_eV is None:
                await self.send_line(
                    writer, ACK_CODE_OK_GENERIC, "Threshold has not been set"
                )
            else:
                msg = (
                    f"Energy setting: {t.incident_energy_eV} eV Settings: {t.gain or 'mid'} gain; "
                    f"threshold: {t.threshold_eV} eV; vcmp: {t.vcmp or 0:.3f} V "
                    f"Trim file: {t.trim_file}"
                )
                await self.send_line(writer, ACK_CODE_OK_GENERIC, msg)
            return
        try:
            energy = int(args.strip())
            if energy <= 0:
                raise ValueError
            threshold = int(energy * 0.5)  # simple heuristic
            t = self.params.threshold
            t.incident_energy_eV = energy
            t.threshold_eV = threshold
            t.gain = "mid"
            t.vcmp = round(random.uniform(0.400, 0.900), 3)
            t.trim_file = f"/tmp/trim_energy{energy}_thr{threshold}.bin"
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"Setting the energy: {t.trim_file}"
            )
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid energy value")

    async def cmd_k(self, args: str, writer: asyncio.StreamWriter):
        last = await self.abort_series()
        await self.send_line(writer, ACK_CODE_INTERRUPT, "ERR kill")
        if last:
            await self.send_line(writer, ACK_CODE_IMAGE_DONE, last)

    async def cmd_camcmd(self, args: str, writer: asyncio.StreamWriter):
        # Only implemented: camcmd K
        sub = args.strip()
        if sub.upper() == "K":
            await self.cmd_k("", writer)
        else:
            await self.send_line(writer, ERROR_CODE, "ERR unsupported camcmd")

    async def cmd_ld_badpix(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            msg = self.params.badpix_map or "none"
            await self.send_line(writer, ACK_CODE_OK_GENERIC, msg)
            return
        if args.strip() in ("0", "off"):
            self.params.badpix_map = None
            await self.send_line(writer, ACK_CODE_OK_GENERIC, "none")
        else:
            self.params.badpix_map = args.strip()
            await self.send_line(writer, ACK_CODE_OK_GENERIC, self.params.badpix_map)

    async def cmd_ld_flatfield(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            msg = self.params.flatfield_file or "none"
            await self.send_line(writer, ACK_CODE_OK_GENERIC, msg)
            return
        if args.strip() in ("0", "off"):
            self.params.flatfield_file = None
            await self.send_line(writer, ACK_CODE_OK_GENERIC, "none")
        else:
            self.params.flatfield_file = args.strip()
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, self.params.flatfield_file
            )

    async def cmd_ratecorr_lutdir(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            msg = (
                f"RateCorrLUTDirectory is {self.params.ratecorr_dir or 'off'}"
                if self.params.ratecorr_dir
                else "RateCorrLUTDirectory is off"
            )
            await self.send_line(writer, ACK_CODE_OK_GENERIC, msg)
            return
        if args.strip() in ("0", "off"):
            self.params.ratecorr_dir = None
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, "Disabling LUT based correction"
            )
        else:
            self.params.ratecorr_dir = args.strip()
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"RateCorrLUTDirectory is {self.params.ratecorr_dir}",
            )

    async def cmd_readout_time(self, args: str, writer: asyncio.StreamWriter):
        ms = self.readout_time * 1000.0
        await self.send_line(
            writer, ACK_CODE_OK_GENERIC, f"Detector readout time [ms]: {ms:.3f}"
        )

    async def cmd_set_retrigger(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                "Retrigger mode is set"
                if self.params.retrigger_mode
                else "Retrigger mode is not set",
            )
            return
        val = args.strip()
        if val not in ("0", "1"):
            await self.send_line(writer, ERROR_CODE, "ERR invalid retrigger mode")
            return
        self.params.retrigger_mode = val == "1"
        await self.send_line(
            writer,
            ACK_CODE_OK_GENERIC,
            "Retrigger mode is set"
            if self.params.retrigger_mode
            else "Retrigger mode is not set",
        )

    async def cmd_gap_fill(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"Detector gap-fill is: {self.params.gap_fill}",
            )
            return
        if args.strip() not in ("0", "-1"):
            await self.send_line(writer, ERROR_CODE, "ERR invalid GapFill (0 or -1)")
            return
        self.params.gap_fill = int(args.strip())
        await self.send_line(
            writer, ACK_CODE_OK_GENERIC, f"Detector gap-fill is: {self.params.gap_fill}"
        )

    async def cmd_thread(self, args: str, writer: asyncio.StreamWriter):
        # Provide temperature and humidity (dummy)
        temp = round(random.uniform(18.0, 35.0), 2)
        hum = round(random.uniform(2.0, 40.0), 1)
        await self.send_line(writer, ACK_CODE_THREAD, f"{temp} {hum}")

    async def cmd_ack_int(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"{self.params.ack_interval}"
            )
            return
        try:
            val = int(args)
            if val < 0:
                raise ValueError
            self.params.ack_interval = val
            await self.send_line(writer, ACK_CODE_OK_GENERIC, f"{val}")
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid SetAckInt")

    async def cmd_reset_cam(self, args: str, writer: asyncio.StreamWriter):
        # Reset only minimal parameters
        self.params.exposure_time = DEFAULT_EXPOSURE_TIME
        self.params.exposure_period = DEFAULT_EXPOSURE_PERIOD
        self.params.n_images = 1
        await self.send_line(writer, ACK_CODE_OK_GENERIC, "Camera reset")

    async def cmd_deb_time(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"Debounce time set to: {self.params.debounce_time:.3f} sec",
            )
            return
        try:
            val = float(args)
            if val < 0 or val >= 85:
                raise ValueError
            self.params.debounce_time = val
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, f"Debounce time set to: {val:.3f} sec"
            )
        except ValueError:
            await self.send_line(writer, ERROR_CODE, "ERR invalid debounce time")

    async def cmd_header_string(self, args: str, writer: asyncio.StreamWriter):
        if not args:
            await self.send_line(
                writer, ACK_CODE_OK_GENERIC, self.params.header_string or "none"
            )
            return
        s = args.strip()
        if len(s) > 68:
            s = s[:68]
        self.params.header_string = s
        await self.send_line(writer, ACK_CODE_OK_GENERIC, self.params.header_string)

    async def cmd_exit(self, args: str, writer: asyncio.StreamWriter):
        await self.send_line(writer, ACK_CODE_OK_GENERIC, "Closing connection")
        raise ConnectionResetError  # Force close

    async def cmd_df(self, args: str, writer: asyncio.StreamWriter):
        try:
            usage = shutil.disk_usage(self.img_path)
            free_k = usage.free // 1024
            await self.send_line(
                writer, ACK_CODE_DISKFREE, f"{free_k} 1K blocks available"
            )
        except Exception as e:
            await self.send_line(writer, ERROR_CODE, f"ERR df: {e}")

    async def cmd_exp_end(self, args: str, writer: asyncio.StreamWriter):
        last = self.state.last_image_path or ""
        await self.send_line(writer, ACK_CODE_EXPEND, last)

    async def cmd_cam_setup(self, args: str, writer: asyncio.StreamWriter):
        # Provide a summarized line
        tleft = 0.0
        if self.state.running and self.params.n_images > 0:
            remaining = self.params.n_images - (self.state.current_index + 1)
            tleft = remaining * self.params.exposure_period
        msg = (
            f"Camera definition: PILATUS3; Camera name: SIM; Camera state: "
            f"{'RUNNING' if self.state.running else 'IDLE'}; Target file: "
            f"{getattr(self, '_current_base_name', 'None')}; Time left: {tleft:.3f}; "
            f"Last image: {self.state.last_image_path or 'None'}; Master PID is: {self.pid}; "
            f"Controlling PID is: {self.pid}; Exposure time: {self.params.exposure_time:.3f}; "
            f"Last completed image: {self.state.current_index}; Shutter is: OPEN"
        )
        await self.send_line(writer, ACK_CODE_CAMSETUP, msg)

    async def cmd_telemetry(self, args: str, writer: asyncio.StreamWriter):
        msg = (
            f"Image format: CBF; ExpTime={self.params.exposure_time:.3f}; "
            f"Period={self.params.exposure_period:.3f}; NImages={self.params.n_images}; "
            f"Threshold={self.params.threshold.threshold_eV or 0}eV; "
            f"Energy={self.params.threshold.incident_energy_eV or 0}eV;"
        )
        await self.send_line(writer, ACK_CODE_TELEMETRY, msg)

    async def cmd_reset_module_power(self, args: str, writer: asyncio.StreamWriter):
        secs = 1.0
        if args.strip():
            try:
                secs = float(args.strip())
            except ValueError:
                pass
        await self.send_line(
            writer,
            ACK_CODE_OK_GENERIC,
            f"Resetting module power, sleeping for {secs:.1f} seconds",
        )
        await asyncio.sleep(secs)
        self.params.threshold = ThresholdSettings()
        await self.send_line(
            writer, ACK_CODE_OK_GENERIC, ">>> Threshold settings no longer valid"
        )

    async def cmd_version(self, args: str, writer: asyncio.StreamWriter):
        await self.send_line(writer, ACK_CODE_VERSION, self.version_string)

    async def cmd_show_pid(self, args: str, writer: asyncio.StreamWriter):
        await self.send_line(writer, ACK_CODE_PID, str(self.pid))

    # --------- Helper / Debug Commands ---------

    async def cmd_help(self, args: str, writer: asyncio.StreamWriter):
        names = " ".join(sorted(self.commands.keys()))
        await self.send_line(writer, ACK_CODE_OK_GENERIC, f"Commands: {names}")

    async def cmd_state(self, args: str, writer: asyncio.StreamWriter):
        async with self.lock:
            s = self.state
            await self.send_line(
                writer,
                ACK_CODE_OK_GENERIC,
                f"running={s.running} armed={s.armed} ext={s.external_mode} "
                f"idx={s.current_index} last={s.last_image_path}",
            )

    async def cmd_trigger(self, args: str, writer: asyncio.StreamWriter):
        if not self.state.external_mode:
            await self.send_line(writer, ERROR_CODE, "ERR not in external mode")
            return
        await self._inject_trigger(single=True)
        await self.send_line(writer, ACK_CODE_OK_GENERIC, "Trigger accepted")

    # ---------- Server & Connection Handling ----------

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr = writer.get_extra_info("peername")
        self.clients.append(writer)
        try:
            await self.info(
                writer, "Pilatus 3 Camserver Simulator (type HELP for list)"
            )
            await self.info(writer, f"Connected from {addr}")
            await self.info(writer, f"Terminator set to repr:{repr(self.terminator)}")
            while True:
                raw_line = await self._read_command(reader)
                if raw_line is None:
                    break
                raw = raw_line.strip()
                if not raw:
                    continue
                parts = raw.split(maxsplit=1)
                cmd = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                handler = self.commands.get(cmd)
                if not handler:
                    await self.send_line(
                        writer, ERROR_CODE, f"ERR UnknownCommand {cmd}"
                    )
                    continue
                try:
                    await handler(args, writer)
                except ConnectionResetError:
                    break
                except Exception as e:
                    self.logger.exception("Error handling command %s", cmd)
                    await self.send_line(
                        writer, ERROR_CODE, f"ERR Exception {type(e).__name__}"
                    )
        finally:
            if writer in self.clients:
                self.clients.remove(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            self.logger.info("Client disconnected %s", addr)

    async def _read_command(self, reader: asyncio.StreamReader) -> Optional[str]:
        """Read bytes until the configured terminator sequence is encountered.

        Returns the decoded string without the terminator. If EOF occurs and no
        bytes were read, returns None. If EOF with partial data, returns that
        partial data (mirrors typical line-based behavior).
        """
        term = self._terminator_bytes
        buf = bytearray()
        term_len = len(term)
        while True:
            chunk = await reader.read(1)
            if not chunk:  # EOF
                if not buf:
                    return None
                else:
                    return buf.decode(errors="replace")
            buf.extend(chunk)
            if len(buf) >= term_len and buf[-term_len:] == term:
                # Strip terminator
                return buf[:-term_len].decode(errors="replace")

    async def start(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
        self.logger.info("Listening on %s", addrs)
        async with server:
            await server.serve_forever()


# ------------------ Argument Parsing & Main ------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Pilatus 3 Camserver Simulator")
    ap.add_argument("--xsize", type=int, default=487, help="Sensor width in pixels")
    ap.add_argument("--ysize", type=int, default=195, help="Sensor height in pixels")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument(
        "--manual-triggers",
        action="store_true",
        help="Require manual TRIGGER command for external modes",
    )
    ap.add_argument(
        "--readout-time",
        type=float,
        default=DEFAULT_READOUT_TIME,
        help="Simulated readout time (sec)",
    )
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "--terminator", default="\\n", help="Line terminator for commands and responses"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sim = CamserverSimulator(
        xsize=args.xsize,
        ysize=args.ysize,
        host=args.host,
        port=args.port,
        terminator=args.terminator,
        manual_triggers=args.manual_triggers,
        readout_time=args.readout_time,
        loop=loop,
    )

    def shutdown():
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass  # Windows

    try:
        loop.run_until_complete(sim.start())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
