"""Step 3 - acquire one image at the configured scan size (sets the scan configuration only;
no beam, HV or stage change).

Answers: is the buffer a BMP ('BM' magic) or raw pixels? row order correct (compare with
the ESPRIT GUI)? pixel size in TRTImageInfoEx vs config?
Saves the raw buffer too, so decode_image can be fixed offline if needed.
"""

import ctypes

import tifffile

from .. import bruker_api as ba
from ._common import connected, parse


def main():
    args, cfg = parse(__doc__)
    w, h = cfg.image_px
    with connected(cfg) as scope:
        scope._call("ImageSetConfiguration", scope.cid, w, h, cfg.image_average, True, False)
        size = ctypes.c_int32(w * h + 4096)
        buf = ctypes.create_string_buffer(size.value)
        info = ba.TRTImageInfoEx()
        scope._call(
            "ImageAquireImage", scope.cid, 1, True, buf, ctypes.byref(size), ctypes.byref(info)
        )

    raw = buf.raw[: size.value]
    with open(f"{args.out}/step3_raw.bin", "wb") as f:
        f.write(raw)
    print(f"returned {size.value} bytes (W*H = {w * h}), magic={raw[:2]!r}")
    print(
        f"info: mag={info.Magnification} px=({info.PixelSizeX:.5f}, {info.PixelSizeY:.5f}) um "
        f"HV={info.HighVoltage} kV WD={info.WorkingDistance} mm  (config {cfg.pixel_size_um} um)"
    )
    img = ba.decode_image(raw, w, h)
    tifffile.imwrite(f"{args.out}/step3_image.tif", img)
    print(
        f"decoded {img.shape}, range {img.min():.3f}..{img.max():.3f} -> step3_image.tif; "
        "compare orientation with the ESPRIT GUI"
    )


if __name__ == "__main__":
    main()
