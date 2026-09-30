"""Step 2 - read-only: SEM data, stage position/range/state, scan configuration, field width.

Answers: stage units really mm? which GetSEMStageState value means idle? does
field_width / image width match the expected pixel size at the current magnification?
Nothing is written to the instrument.
"""

import ctypes
import json
from pathlib import Path

from ._common import connected, parse


def main():
    args, cfg = parse(__doc__)
    with connected(cfg) as scope:
        w, h, avg = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
        ch1, ch2 = ctypes.c_bool(), ctypes.c_bool()
        scope._call("ImageGetConfiguration", scope.cid, *map(ctypes.byref, (w, h, avg, ch1, ch2)))
        fw = scope.field_width_um()
        state = {
            "sem": scope.sem_data(),
            "stage_mm_deg": scope.stage_position_mm(),
            "stage_range": scope.stage_range_mm(),
            "stage_state": scope.stage_state(),
            "image_config": {
                "w": w.value,
                "h": h.value,
                "average": avg.value,
                "ch1": ch1.value,
                "ch2": ch2.value,
            },
            "field_width_um": fw,
            "pixel_size_um_at_config_width": fw / cfg.image_px[0],
        }
    print(json.dumps(state, indent=2))
    (Path(args.out) / "step2_state.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
