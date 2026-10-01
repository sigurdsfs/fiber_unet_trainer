"""ctypes bindings for the Bruker ESPRIT API (Bruker.API.Esprit64.dll, API v2.7.0.0).

Source of truth: the C++ headers shipped on the Bruker USB stick
(E:\\Esprit API\\C++\\Interfaces\\Bruker.API.{Types,Common,Esprit}.h). Only the functions
the SEM->segment->EDS loop needs are bound, and deliberately NOT the ones that change the
beam or shut ESPRIT down (SetSEMData, SetSEMProbeCurrent, CloseClient, ...).

Conventions taken from the headers:
  * one DLL exports both the "Common" and "Esprit" functions (LoadEspritAPI calls
    LoadCommonFunctions on the same file);
  * calling convention is WINAPI/stdcall -> ctypes.WinDLL;
  * every struct is `#pragma pack(1)` -> `_pack_ = 1`;
  * every function returns int32: 0 = ok, 1 = warning (wrong element), -201 = ok
    (client was already open), any other negative value = error.

This module has no dependency on the DLL at import time, so it is importable (and its
structs testable) on any machine.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_double,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_void_p,
)

# ---------------------------------------------------------------------------------
# Result codes (Bruker.API.Types.h)
# ---------------------------------------------------------------------------------
OK_CODES = {0, 1, -201}

ERROR_NAMES = {
    1: "IFC_WARNING_WRONG_ELEMENT",
    -1: "IFC_ERROR_IN_EXECUTION",
    -2: "IFC_ERROR_WRONG_PARAMETER",
    -3: "IFC_ERROR_SPECTRUM_BUFFER_EMPTY",
    -4: "IFC_ERROR_PARAMETER_MISSED",
    -5: "IFC_ERROR_TOO_MANY_PARAMETERS",
    -6: "IFC_ERROR_USER_TERMINATED",
    -7: "IFC_ERROR_TIMEOUT",
    -8: "IFC_ERROR_UNKNOWN_VALUE_NAME",
    -9: "IFC_ERROR_WRONG_VALUE_TYPE",
    -10: "IFC_ERROR_WRONG_LICENCE",
    -11: "IFC_ERROR_RESULT_BUFFER_INSUFFICIENT",
    -12: "IFC_ERROR_HARDWARE_LOCKED",
    -13: "IFC_ERROR_SEQUENCE",
    -21: "CONN_ERROR_UNKNOWN",
    -22: "CONN_ERROR_INTERFACE_NOT_CONNECTED",
    -23: "CONN_ERROR_PARAMETER_MISSED",
    -24: "CONN_ERROR_ANSWER_TIMEOUT",
    -25: "CONN_ERROR_SERVER_NOT_RESPONDING",
    -26: "CONN_ERROR_RESULT_MISSED",
    -27: "CONN_ERROR_NO_INTERFACE",
    -28: "CONN_ERROR_INVALID_LOGIN",
    -29: "CONN_ERROR_NO_CONNECTION_TO_SERVER",
    -30: "CONN_ERROR_SERVER",
    -31: "CONN_ERROR_ACTION_ABORTED",
    -51: "IFS_ERROR_PARAMETER_MISSED",
    -52: "IFS_ERROR_FUNCTION_NOT_IMPLEMENTED",
    -53: "IFS_ERROR_FUNTION_EXCEPTED",
    -101: "ERROR_WRONG_PARAMETER",
    -102: "ERROR_FILE_NOT_EXIST",
    -103: "ERROR_NO_CONNECTION",
    -104: "ERROR_NO_ANSWER",
    -105: "ERROR_CAN_NOT_START_PROCESS",
    -106: "ERROR_INVALID_RESULT_DATA",
    -107: "ERROR_SETTINGS_NOT_FOUND",
    -108: "ERROR_NO_SERVER_CONNECTION",
    -109: "ERROR_IN_EXECUTION",
    -110: "ERROR_IFC_BUSY",
    -201: "STATE_WAS_RUNNING_BEFORE",
}


class BrukerAPIError(RuntimeError):
    def __init__(self, function: str, code: int, detail: str = ""):
        self.function, self.code = function, code
        name = ERROR_NAMES.get(code, "unknown")
        super().__init__(f"{function} failed: {code} ({name}){' - ' + detail if detail else ''}")


# ---------------------------------------------------------------------------------
# Structs (all pack(1))
# ---------------------------------------------------------------------------------
class TSegment(ctypes.Structure):
    """One scan-line run. A single EDS point is a segment with XCount = 1."""

    _pack_ = 1
    _fields_ = [("Y", c_uint32), ("XStart", c_uint32), ("XCount", c_int32)]


class TRTAPISpectrumHeaderRec(ctypes.Structure):
    """Header at the start of every spectrum buffer; channel data follows it."""

    _pack_ = 1
    _fields_ = [
        ("IdentifierLength", c_uint8),
        ("Identifier", ctypes.c_char * 25),  # 'Bruker XRay spectrum'
        ("Version", c_int32),
        ("Size", c_int32),  # size in bytes
        ("DateTime", c_double),  # Delphi TDateTime
        ("ChannelCount", c_int32),
        ("ChannelOffset", c_int32),  # first channel index
        ("CalibrationAbs", c_double),  # keV of first channel
        ("CalibrationLin", c_double),  # keV per channel
        ("SigmaAbs", c_double),
        ("SigmaLin", c_double),
        ("RealTime", c_int32),  # ms
        ("LifeTime", c_int32),  # ms
    ]


class TRTImageInfoEx(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("Magnification", c_int32),
        ("PixelSizeX", c_double),  # um per pixel
        ("PixelSizeY", c_double),
        ("HighVoltage", c_double),  # kV
        ("WorkingDistance", c_double),  # mm
    ]


class TOpenClientOptions(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("Version", c_int32),
        ("GUIMode", c_int32),
        ("StartNew", c_bool),
        ("IdentifierLength", c_uint8),
        ("TCPHost", ctypes.c_char * 64),  # from version 2
        ("TCPPort", c_uint16),  # from version 2
    ]


# ---------------------------------------------------------------------------------
# Function signatures: name -> argtypes (restype is always int32)
# ---------------------------------------------------------------------------------
_u32p, _i32p, _dp, _bp = POINTER(c_uint32), POINTER(c_int32), POINTER(c_double), POINTER(c_bool)

SIGNATURES: dict[str, list] = {
    # connection
    "QueryServers": [c_char_p, c_int32],
    "OpenClient": [c_char_p, c_char_p, c_char_p, c_bool, c_bool, _u32p],
    "OpenClientTCP": [c_char_p, c_char_p, c_char_p, c_char_p, c_uint16, TOpenClientOptions, _u32p],
    "QueryInfo": [c_uint32, c_char_p, c_int32],
    "CheckConnection": [c_uint32],
    "CloseConnection": [c_uint32],  # NOT CloseClient: that shuts ESPRIT down
    "GetDebugErrorString": [c_uint32, c_int32, c_char_p, _i32p],
    # microscope (read) + stage
    "GetSEMData": [c_uint32, _dp, _dp, _dp],  # mag, HV (kV), WD (mm)
    "GetSEMStageData": [c_uint32, _dp, _dp, _dp, _dp, _dp],  # x, y, z (mm), tilt, rot (deg)
    "SetSEMStageData": [c_uint32, c_double, c_double, c_double, c_double, c_double],
    "GetSEMStageRange": [c_uint32] + [_dp] * 10,
    "GetSEMStageState": [c_uint32, _i32p],  # meaning undocumented
    # image
    "ImageGetConfiguration": [c_uint32, _u32p, _u32p, _u32p, _bp, _bp],
    "ImageSetConfiguration": [c_uint32, c_uint32, c_uint32, c_uint32, c_bool, c_bool],
    "ImageAquireImage": [c_uint32, c_int32, c_bool, c_void_p, _i32p, POINTER(TRTImageInfoEx)],
    "ImageGetFieldWidth": [c_uint32, _dp],
    # EDS
    "GetSpectrumMeasureState": [c_uint32, c_int32, _bp, _dp, _dp],
    "StartPointListMeasurement": [c_uint32, c_int32, c_uint32, POINTER(TSegment), c_uint32],
    "GetPointListSpectrum": [c_uint32, c_int32, POINTER(TRTAPISpectrumHeaderRec), _i32p],
    "SavePointListSpectrum": [c_uint32, c_int32, c_char_p],
    "QuantifyPointListSpectrum": [
        c_uint32,
        c_int32,
        c_char_p,
        c_char_p,
        c_char_p,
        c_int32,
        POINTER(TRTAPISpectrumHeaderRec),
        c_int32,
    ],
    "GetQuantificationMethods": [c_uint32, c_bool, c_char_p, c_int32],
}


class BrukerAPI:
    """Thin wrapper: loads the DLL, sets argtypes/restype for every bound function, and
    raises BrukerAPIError on non-OK result codes. No state beyond the DLL handle."""

    def __init__(self, dll_path: str):
        if ctypes.sizeof(c_void_p) != 8 and dll_path.lower().endswith("64.dll"):
            raise RuntimeError("64-bit DLL needs 64-bit Python (and vice versa).")
        self.dll = ctypes.WinDLL(dll_path)
        for name, argtypes in SIGNATURES.items():
            fn = getattr(self.dll, name)  # AttributeError here = header/DLL mismatch
            fn.argtypes = argtypes
            fn.restype = c_int32

    def call(self, name: str, *args, cid: int | None = None) -> int:
        code = getattr(self.dll, name)(*args)
        if code not in OK_CODES:
            raise BrukerAPIError(
                name, code, self.error_string(cid, code) if cid is not None else ""
            )
        return code

    def error_string(self, cid: int, code: int) -> str:
        buf, size = ctypes.create_string_buffer(512), c_int32(512)
        try:
            if self.dll.GetDebugErrorString(cid, code, buf, ctypes.byref(size)) == 0:
                return buf.value.decode(errors="replace")
        except OSError:
            pass
        return ""


# ---------------------------------------------------------------------------------
# Buffer helpers (pure Python, unit-tested)
# ---------------------------------------------------------------------------------
def points_to_segments(points_xy) -> ctypes.Array:
    """(N, 2) pixel coords (x=col, y=row) -> TSegment array with XCount = 1 each."""
    arr = (TSegment * len(points_xy))()
    for i, (x, y) in enumerate(points_xy):
        arr[i].Y, arr[i].XStart, arr[i].XCount = int(y), int(x), 1
    return arr


def parse_spectrum(buf: bytes):
    """Spectrum buffer -> (header, counts, (offset_keV, keV_per_channel)).

    TODO(bring-up step 4): the headers do not document the channel data layout. Assumed:
    int32 counts, one per channel, directly after the 90-byte header. Verify by comparing
    against the same spectrum saved as .spx via SavePointListSpectrum.
    """
    import numpy as np

    hdr = TRTAPISpectrumHeaderRec.from_buffer_copy(buf[: ctypes.sizeof(TRTAPISpectrumHeaderRec)])
    start = ctypes.sizeof(TRTAPISpectrumHeaderRec)
    counts = np.frombuffer(buf, dtype="<i4", count=hdr.ChannelCount, offset=start).astype(
        np.float32
    )
    return hdr, counts, (hdr.CalibrationAbs, hdr.CalibrationLin)


def decode_image(buf: bytes, width: int, height: int):
    """ImageAquireImage buffer -> float32 (H, W) in [0, 1].

    The API returns W*H 8-bit pixels plus a header; the samples write it straight to a .bmp,
    so a Windows BMP is expected. Falls back to "last W*H bytes" if there is no 'BM' magic.
    TODO(bring-up step 3): confirm which branch the real instrument takes (and row order).
    """
    import io

    import numpy as np
    from PIL import Image

    if buf[:2] == b"BM":
        img = np.asarray(Image.open(io.BytesIO(buf)).convert("L"))
    else:
        img = np.frombuffer(buf[-width * height :], dtype=np.uint8).reshape(height, width)
    if img.shape != (height, width):
        raise ValueError(f"image is {img.shape}, expected {(height, width)}")
    return img.astype(np.float32) / 255.0
