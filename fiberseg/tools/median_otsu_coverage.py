# median_otsu_coverage.py
"""Classical median-filter + Otsu-threshold coverage estimate for every image in a config.

For each image referenced by `data.images_dir`/`data.image_glob` (via `find_pairs`, so it's
the same 299-image set used by `evaluate_predictions.py`/`merge_sample_overview.py`), this:

1. Median-filters the image, then uses 3-class Otsu (`threshold_multiotsu`) to separate the
   image into filter pore holes (darkest), flat filter substrate (mid), and particles/fibers
   (brightest) -- these SEM filter images have bright charging-artifact halos ringing each
   pore hole, so the dark-pore region is dilated by `--hole-dilation-radius` pixels and
   excluded before re-thresholding the remaining pixels, otherwise those halos get counted
   as "foreground" too.
2. Saves the resulting particle mask as a .tif in `--mask-out-dir` so it can be checked by eye
   -- tune `--hole-dilation-radius` per dataset/magnification if halos are still visible in
   the saved masks, or if too much real particle area near a pore gets excluded.
3. Computes what fraction of the image the mask covers (as a percentage) and inserts it into
   an existing metrics CSV as a new column placed immediately next to "Filter Coverage (%)",
   for side-by-side comparison against the ground-truth-reported value.

Since real filter samples also collect particles other than the fiber of interest, this
coverage is expected to run higher than the ground-truth fiber mask's foreground fraction --
that's normal, not a bug.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_multiotsu, threshold_otsu

from ..config import load_config
from ..dataset import _normalize_image, _read_gray, find_pairs

NEW_COLUMN = "Median Otsu Coverage (%)"


def median_otsu_mask(
    img: np.ndarray,
    median_radius: int = 4,
    num_passes: int = 4,
    hole_dilation_radius: int = 50,
) -> np.ndarray:
    """Iteratively median-filter `img`, then isolate particles via hole-aware Otsu splitting.

    Uses cv2.medianBlur (histogram-based, only supports uint8 for kernel sizes > 5) rather
    than scipy.ndimage.median_filter, which is orders of magnitude slower on the large SEM
    images this project works with.
    """
    size = 2 * median_radius + 1
    filtered = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    for _ in range(num_passes):
        filtered = cv2.medianBlur(filtered, size)

    hole_low, _ = threshold_multiotsu(filtered, classes=3)
    hole_mask = filtered <= hole_low

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * hole_dilation_radius + 1, 2 * hole_dilation_radius + 1)
    )
    excluded = cv2.dilate(hole_mask.astype(np.uint8), kernel).astype(bool)

    threshold = threshold_otsu(filtered[~excluded])
    return (filtered > threshold) & ~excluded


def main():
    parser = argparse.ArgumentParser(
        description="Compute a median+Otsu foreground mask for every image in a config, save "
        "the masks for visual QC, and add a coverage-fraction column to a metrics CSV."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with an 'image' column to augment (e.g. metrics_with_sample_info.csv).",
    )
    parser.add_argument(
        "--mask-out-dir",
        required=True,
        help="Folder to save median_otsu masks in, for visual verification.",
    )
    parser.add_argument("--median-radius", type=int, default=4)
    parser.add_argument("--num-passes", type=int, default=4)
    parser.add_argument(
        "--hole-dilation-radius",
        type=int,
        default=50,
        help="Pixels to expand each detected pore hole by before excluding it, to also cover "
        "its bright charging-artifact rim (default: 50; tune based on saved masks).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: overwrite --csv in place).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs = find_pairs(cfg.data)

    mask_out_dir = Path(args.mask_out_dir)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    coverage_by_filename: dict[str, float] = {}

    for i, pair in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] {pair.image_path.name}")

        img = _normalize_image(_read_gray(pair.image_path))
        mask = median_otsu_mask(
            img,
            median_radius=args.median_radius,
            num_passes=args.num_passes,
            hole_dilation_radius=args.hole_dilation_radius,
        )

        coverage_by_filename[pair.image_path.name] = 100.0 * float(mask.mean())

        out_mask_path = mask_out_dir / f"{pair.image_path.stem}_otsu_mask.tif"
        tifffile.imwrite(out_mask_path, (mask.astype(np.uint8) * 255))

    df = pd.read_csv(args.csv)
    df[NEW_COLUMN] = df["image"].map(coverage_by_filename)

    if "Filter Coverage (%)" in df.columns:
        cols = [c for c in df.columns if c != NEW_COLUMN]
        insert_at = cols.index("Filter Coverage (%)") + 1
        cols.insert(insert_at, NEW_COLUMN)
        df = df[cols]
    else:
        print("Warning: 'Filter Coverage (%)' column not found in --csv; appended new column at the end.")

    unmatched = df[NEW_COLUMN].isna().sum()
    if unmatched:
        print(f"Warning: {unmatched} rows in --csv had no matching computed image.")

    out_path = Path(args.out) if args.out else Path(args.csv)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Saved {len(pairs)} median_otsu masks to {mask_out_dir}")


if __name__ == "__main__":
    main()
