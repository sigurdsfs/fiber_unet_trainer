"""Step 5 - one complete tile WITHOUT stage movement: image -> segment -> targets -> point list.

Uses the real segmenter (settings.ckpt) and the settings validated in step 4. If
settings.quant_method is set, also captures one ESPRIT quant result to quant_example.txt
so classify.parse_quant_result can be written.
"""

import json
from pathlib import Path

import numpy as np
import tifffile

from ..segmenter import FiberSegSegmenter
from ..targeting import build_point_list, find_targets
from ._common import confirm, connected, parse


def main():
    args, cfg = parse(__doc__)
    out = Path(args.out)
    segmenter = FiberSegSegmenter(cfg)
    with connected(cfg) as scope:
        scope.setup(cfg)
        img = scope.acquire_image()
        labels = segmenter(img)
        fibers = find_targets(labels, cfg, np.random.default_rng(cfg.seed))
        pts_rc, owner = build_point_list(fibers)
        tifffile.imwrite(out / "step5_image.tif", img)
        np.save(out / "step5_labels.npy", labels)
        n_countable = sum(f.countable for f in fibers)
        print(f"{labels.max()} objects, {n_countable} countable, {len(pts_rc)} EDS points")
        if not pts_rc:
            return
        confirm(
            f"Measure {len(pts_rc)} points at {cfg.eds_real_time_ms} ms each "
            f"(~{len(pts_rc) * cfg.eds_real_time_ms / 1000:.0f} s)?"
        )
        spectra, calib = scope.acquire_point_spectra(np.array([(c, r) for r, c in pts_rc]))
        np.save(out / "step5_spectra.npy", spectra)
        (out / "step5_points.json").write_text(
            json.dumps({"points_rc": pts_rc, "owner": owner, "calib": calib})
        )
        if cfg.quant_method:
            (out / "quant_example.txt").write_text(scope.quantify_point(0, cfg.quant_method))
    print("done - check that spectra on fibres differ from their background points")


if __name__ == "__main__":
    main()
