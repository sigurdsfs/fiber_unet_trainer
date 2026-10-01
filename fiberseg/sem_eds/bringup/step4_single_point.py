"""Step 4 - EDS on 2-3 operator-chosen pixels of a freshly acquired frame.

    python -m fiberseg.sem_eds.bringup.step4_single_point --points 300,200 500,410

Answers, and writes the answers into settings:
  * does StartPointListMeasurement block?      -> point_list_wait
  * GetPointListSpectrum index base 0 or 1?    -> point_list_index_base
  * spectrum layout after the 90-byte header   -> bruker_api.parse_spectrum
    (compare with the .spx saved by SavePointListSpectrum, opened in ESPRIT)
"""

import ctypes
import time

import numpy as np

from .. import bruker_api as ba
from ._common import confirm, connected, parse


def main():
    args, cfg = parse(
        __doc__, lambda ap: ap.add_argument("--points", nargs="+", required=True, help="x,y pixel")
    )
    pts = np.array([tuple(map(int, p.split(","))) for p in args.points])
    confirm(f"EDS on {len(pts)} points, {cfg.eds_real_time_ms} ms each (real time).")

    with connected(cfg) as scope:
        scope.setup(cfg)
        scope.acquire_image()  # defines the frame the points refer to
        segs = ba.points_to_segments(pts)

        t0 = time.time()
        scope._call(
            "StartPointListMeasurement", scope.cid, cfg.spu, len(pts), segs, cfg.eds_real_time_ms
        )
        dt = time.time() - t0
        expected = len(pts) * cfg.eds_real_time_ms / 1000
        blocking = dt >= expected * 0.9
        print(f"StartPointListMeasurement returned after {dt:.1f} s (needs >= {expected:.1f} s)")
        print("-> BLOCKING" if blocking else "-> ASYNC: set point_list_wait to poll_state or sleep")

        if not blocking:
            # Does the spectrum measure state reflect the point list? (decides poll_state vs sleep)
            for _ in range(int(expected * 3) + 10):
                running, st, rate = ctypes.c_bool(), ctypes.c_double(), ctypes.c_double()
                scope._call(
                    "GetSpectrumMeasureState",
                    scope.cid,
                    cfg.spu,
                    ctypes.byref(running),
                    ctypes.byref(st),
                    ctypes.byref(rate),
                )
                print(f"  running={running.value} state={st.value:.0f}% rate={rate.value:.0f} cps")
                if not running.value:
                    break
                time.sleep(1)

        for idx in range(0, len(pts) + 1):  # probe both index bases
            size = ctypes.c_int32(cfg.spectrum_buf_bytes)
            buf = ctypes.create_string_buffer(size.value)
            try:
                scope._call(
                    "GetPointListSpectrum",
                    scope.cid,
                    idx,
                    ctypes.cast(buf, ctypes.POINTER(ba.TRTAPISpectrumHeaderRec)),
                    ctypes.byref(size),
                )
            except ba.BrukerAPIError as e:
                print(f"index {idx}: {e}")
                continue
            hdr, counts, calib = ba.parse_spectrum(buf.raw)
            print(
                f"index {idx}: {size.value} B, id={hdr.Identifier!r}, channels={hdr.ChannelCount}, "
                f"calib={calib}, real={hdr.RealTime} ms, live={hdr.LifeTime} ms, "
                f"total counts={counts.sum():.0f}"
            )
            with open(f"{args.out}/step4_spec_{idx}.bin", "wb") as f:
                f.write(buf.raw[: size.value])
            scope._call(
                "SavePointListSpectrum", scope.cid, idx, f"bringup_point_{idx}.spx".encode()
            )


if __name__ == "__main__":
    main()
