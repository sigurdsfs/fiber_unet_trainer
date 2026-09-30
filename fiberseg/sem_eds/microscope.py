"""Microscope backends behind one interface, so --dry-run always works.

  Microscope          - interface the pipeline uses
  MockMicroscope      - simulated fibres + spectra (from the handoff, unchanged behaviour)
  QuantaxMicroscope   - real ESPRIT API via bruker_api.BrukerAPI

Pixel coordinates passed to acquire_point_spectra are (x=col, y=row) in the frame of the
LAST acquired image.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np
from scipy import ndimage as ndi
from skimage import draw, morphology

from . import bruker_api as ba
from .settings import SemEdsConfig

LINES_KEV = {
    "O": 0.525,
    "Na": 1.041,
    "Mg": 1.254,
    "Al": 1.487,
    "Si": 1.740,
    "Ca": 3.692,
    "Fe": 6.404,
}


class Microscope:
    def connect(self) -> None: ...
    def setup(self, cfg: SemEdsConfig) -> None: ...
    def move_stage(self, x_um: float, y_um: float) -> None: ...
    def acquire_image(self) -> np.ndarray: ...  # float32 (H, W) in [0, 1]

    def acquire_point_spectra(
        self, points_xy: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """-> spectra (N, channels), energy calibration (offset_keV, keV_per_channel)."""
        ...

    def close(self) -> None: ...


# =====================================================================================
# Real instrument
# =====================================================================================
class StageLimitError(RuntimeError):
    pass


class QuantaxMicroscope(Microscope):
    def __init__(self, cfg: SemEdsConfig):
        self.cfg = cfg
        self.api: ba.BrukerAPI | None = None
        self.cid = ctypes.c_uint32(0)
        self.last_info: ba.TRTImageInfoEx | None = None

    def _call(self, name, *args):
        return self.api.call(name, *args, cid=self.cid.value)

    # --- connection -------------------------------------------------------------------
    def connect(self):
        c = self.cfg
        self.api = ba.BrukerAPI(c.dll_path)
        srv, usr, pwd = c.server.encode("cp1252"), c.user.encode(), c.password.encode()
        if c.tcp_host:
            opts = ba.TOpenClientOptions(
                Version=2, GUIMode=1, StartNew=False
            )  # TODO: GUIMode values undocumented
            self._call(
                "OpenClientTCP",
                srv,
                usr,
                pwd,
                c.tcp_host.encode(),
                c.tcp_port,
                opts,
                ctypes.byref(self.cid),
            )
        else:
            # StartNew=False (attach to a running ESPRIT), GUI=True (operator can watch)
            self._call("OpenClient", srv, usr, pwd, False, True, ctypes.byref(self.cid))

    def query_info(self) -> str:
        buf = ctypes.create_string_buffer(4096)
        self._call("QueryInfo", self.cid, buf, len(buf))
        return buf.value.decode("cp1252", errors="replace")

    def close(self):
        if self.api is not None and self.cid.value:
            try:
                self._call("CloseConnection", self.cid)  # ESPRIT keeps running
            except ba.BrukerAPIError as e:
                print(f"warning: {e}")
            self.cid = ctypes.c_uint32(0)

    # --- read-only state --------------------------------------------------------------
    def sem_data(self) -> dict:
        mag, hv, wd = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        self._call("GetSEMData", self.cid, ctypes.byref(mag), ctypes.byref(hv), ctypes.byref(wd))
        return {"magnification": mag.value, "hv_kv": hv.value, "wd_mm": wd.value}

    def stage_position_mm(self) -> tuple[float, float, float, float, float]:
        v = [ctypes.c_double() for _ in range(5)]
        self._call("GetSEMStageData", self.cid, *map(ctypes.byref, v))
        return tuple(x.value for x in v)  # x, y, z (mm), tilt, rot (deg)

    def stage_range_mm(self) -> dict:
        v = [ctypes.c_double() for _ in range(10)]
        self._call("GetSEMStageRange", self.cid, *map(ctypes.byref, v))
        keys = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "tmin", "tmax", "rmin", "rmax"]
        return dict(zip(keys, (x.value for x in v)))

    def stage_state(self) -> int:
        s = ctypes.c_int32()
        self._call("GetSEMStageState", self.cid, ctypes.byref(s))
        return s.value  # TODO(step 2/6): which value = "idle"?

    def field_width_um(self) -> float:
        fw = ctypes.c_double()
        self._call("ImageGetFieldWidth", self.cid, ctypes.byref(fw))
        return fw.value

    # --- setup ------------------------------------------------------------------------
    def setup(self, cfg: SemEdsConfig):
        """Configure the scan only. Magnification/HV/WD are set by the operator in the SEM
        software; this checks the resulting pixel size instead of changing the beam."""
        self.cfg = cfg
        w, h = cfg.image_px
        self._call("ImageSetConfiguration", self.cid, w, h, cfg.image_average, True, False)
        px = self.field_width_um() / w
        self._check_pixel_size(px, "ImageGetFieldWidth")
        self._stage_range = self.stage_range_mm()

    def _check_pixel_size(self, px_um: float, source: str):
        want = self.cfg.pixel_size_um
        if abs(px_um - want) / want > self.cfg.pixel_size_tol:
            raise RuntimeError(
                f"pixel size from {source} is {px_um:.4f} um, config expects {want:.4f} um "
                f"- adjust magnification in the SEM software"
            )

    # --- stage ------------------------------------------------------------------------
    def check_stage_target(self, x_um: float, y_um: float):
        c = self.cfg
        for v, (lo, hi), ax in ((x_um, c.stage_x_limits_um, "x"), (y_um, c.stage_y_limits_um, "y")):
            if not lo < hi:
                raise StageLimitError(f"stage_{ax}_limits_um not set - refusing to move")
            if not lo <= v <= hi:
                raise StageLimitError(f"{ax}={v:.1f} um outside configured limits [{lo}, {hi}]")
        r = getattr(self, "_stage_range", None) or self.stage_range_mm()
        if not (r["xmin"] <= x_um / 1000 <= r["xmax"] and r["ymin"] <= y_um / 1000 <= r["ymax"]):
            raise StageLimitError(f"({x_um:.1f}, {y_um:.1f}) um outside hardware range {r}")

    def move_stage(self, x_um, y_um):
        self.check_stage_target(x_um, y_um)
        _, _, z, tilt, rot = self.stage_position_mm()  # keep Z / tilt / rotation unchanged
        self._call("SetSEMStageData", self.cid, x_um / 1000, y_um / 1000, z, tilt, rot)
        # TODO(step 6): replace position polling with GetSEMStageState once its values are known.
        t0 = time.time()
        while time.time() - t0 < self.cfg.stage_timeout_s:
            x, y, *_ = self.stage_position_mm()
            if abs(x * 1000 - x_um) < 0.5 and abs(y * 1000 - y_um) < 0.5:
                return
            time.sleep(0.2)
        raise TimeoutError(f"stage did not reach ({x_um:.1f}, {y_um:.1f}) um")

    # --- image ------------------------------------------------------------------------
    def acquire_image(self):
        w, h = self.cfg.image_px
        size = ctypes.c_int32(w * h + 4096)  # samples: W*H + ~2000 header bytes
        buf = ctypes.create_string_buffer(size.value)
        info = ba.TRTImageInfoEx()
        self._call(
            "ImageAquireImage", self.cid, 1, False, buf, ctypes.byref(size), ctypes.byref(info)
        )
        self.last_info = info
        self._check_pixel_size(info.PixelSizeX, "image header")
        return ba.decode_image(buf.raw[: size.value], w, h)

    # --- EDS --------------------------------------------------------------------------
    def acquire_point_spectra(self, points_xy):
        c = self.cfg
        segs = ba.points_to_segments(points_xy)
        n = len(points_xy)
        self._call("StartPointListMeasurement", self.cid, c.spu, n, segs, c.eds_real_time_ms)
        self._wait_point_list(n)

        spectra, calib = [], None
        for i in range(n):
            size = ctypes.c_int32(c.spectrum_buf_bytes)
            buf = ctypes.create_string_buffer(size.value)
            hdr_p = ctypes.cast(buf, ctypes.POINTER(ba.TRTAPISpectrumHeaderRec))
            self._call(
                "GetPointListSpectrum",
                self.cid,
                i + c.point_list_index_base,
                hdr_p,
                ctypes.byref(size),
            )
            _, counts, calib_i = ba.parse_spectrum(buf.raw)
            calib = calib or calib_i
            spectra.append(counts)
        return np.stack(spectra), calib

    def _wait_point_list(self, n: int):
        """TODO(step 4): find out whether StartPointListMeasurement blocks. The Bruker sample
        only sleeps 5 s. Pick the mode that matches reality in settings.point_list_wait."""
        c = self.cfg
        budget = n * c.eds_real_time_ms / 1000 * 1.5 + 10
        if c.point_list_wait == "blocking":
            return
        if c.point_list_wait == "sleep":
            time.sleep(budget)
            return
        running, state, rate = ctypes.c_bool(True), ctypes.c_double(), ctypes.c_double()
        t0 = time.time()
        while time.time() - t0 < budget:
            self._call(
                "GetSpectrumMeasureState",
                self.cid,
                c.spu,
                ctypes.byref(running),
                ctypes.byref(state),
                ctypes.byref(rate),
            )
            if not running.value:
                return
            time.sleep(0.5)
        raise TimeoutError("point list measurement did not finish")

    def quantify_point(
        self, index: int, method: str, params: str = "ResultType=quantification\n"
    ) -> str:
        """ESPRIT quantification of one point-list spectrum (0-based index here)."""
        size = self.cfg.spectrum_buf_bytes
        spec = ctypes.create_string_buffer(size)
        result = ctypes.create_string_buffer(8192)
        self._call(
            "QuantifyPointListSpectrum",
            self.cid,
            index + self.cfg.point_list_index_base,
            method.encode(),
            params.encode(),
            result,
            len(result),
            ctypes.cast(spec, ctypes.POINTER(ba.TRTAPISpectrumHeaderRec)),
            size,
        )
        return result.value.decode("cp1252", errors="replace")


# =====================================================================================
# Simulation (handoff MockMicroscope)
# =====================================================================================
MOCK_MINERALS = {
    "chrysotile": {"O": 80, "Mg": 120, "Si": 100, "Fe": 5},
    "amosite": {"O": 80, "Mg": 15, "Si": 100, "Fe": 70},
    "crocidolite": {"O": 80, "Na": 20, "Si": 100, "Fe": 60},
    "tremolite": {"O": 80, "Mg": 50, "Ca": 35, "Si": 100},
    "gypsum": {"O": 60, "Ca": 100},
}
MOCK_SUBSTRATE = {"O": 40, "Si": 3}


class MockMicroscope(Microscope):
    CHANNELS, OFFSET_KEV, GAIN_KEV = 2048, 0.0, 0.01

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self._labels = None
        self._minerals: dict[int, str] = {}
        self._img = None

    def connect(self):
        pass

    def setup(self, cfg):
        self.cfg = cfg

    def move_stage(self, x_um, y_um):
        w, h = self.cfg.image_px
        labels = np.zeros((h, w), np.int32)
        self._minerals = {}
        for i in range(1, self.rng.integers(4, 12) + 1):
            L = self.rng.uniform(40, 450)
            a = self.rng.uniform(0, np.pi)
            r0, c0 = self.rng.uniform(0, h), self.rng.uniform(0, w)
            r1, c1 = r0 + L * np.sin(a), c0 + L * np.cos(a)
            line = np.zeros_like(labels, bool)
            rr, cc = draw.line(
                int(r0), int(c0), int(np.clip(r1, 0, h - 1)), int(np.clip(c1, 0, w - 1))
            )
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            line[rr[ok], cc[ok]] = True
            line = ndi.binary_dilation(
                line, structure=morphology.disk(int(self.rng.integers(1, 4)))
            )
            labels[line] = i
            self._minerals[i] = str(self.rng.choice(list(MOCK_MINERALS)))
        self._labels = labels
        img = 0.2 + 0.6 * (labels > 0) + self.rng.normal(0, 0.05, labels.shape)
        self._img = np.clip(ndi.gaussian_filter(img, 0.7), 0, 1).astype(np.float32)

    def acquire_image(self):
        return self._img.copy()

    def _spectrum(self, comp: dict, live_time_s: float) -> np.ndarray:
        e = self.OFFSET_KEV + self.GAIN_KEV * np.arange(self.CHANNELS)
        s = 40 * np.exp(-e / 4) * (e > 0.2)
        for el, amp in comp.items():
            E = LINES_KEV[el]
            s = s + amp * 20 * self.GAIN_KEV / (0.05 * np.sqrt(2 * np.pi)) * np.exp(
                -0.5 * ((e - E) / 0.05) ** 2
            )
        return self.rng.poisson(s * live_time_s).astype(np.float32)

    def acquire_point_spectra(self, points_xy):
        t = self.cfg.eds_real_time_ms / 1000
        out = []
        for x, y in points_xy:
            lab = self._labels[int(y), int(x)]
            comp = dict(MOCK_SUBSTRATE)
            if lab:
                comp = {k: v * 0.5 for k, v in MOCK_SUBSTRATE.items()}
                for el, a in MOCK_MINERALS[self._minerals[lab]].items():
                    comp[el] = comp.get(el, 0) + a
            out.append(self._spectrum(comp, t))
        return np.stack(out), (self.OFFSET_KEV, self.GAIN_KEV)

    def close(self):
        pass
