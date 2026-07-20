# add_gt_foreground_fraction.py
"""Add the ground-truth foreground (fiber) pixel fraction as a percentage column to a metrics CSV.

For every image referenced by a config (via `find_pairs`), reads its ground-truth mask and
computes what percentage of pixels are foreground (`mask > 0`, the same convention used
everywhere else in this codebase). The result is inserted next to the other coverage columns
("Median Otsu Coverage (%)" / "Filter Coverage (%)") if either is already present in the CSV,
so all coverage estimates read side by side.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..dataset import _read_gray, find_pairs

NEW_COLUMN = "GT Foreground Fraction (%)"


def main():
    parser = argparse.ArgumentParser(
        description="Compute the fraction of each ground-truth mask that is foreground (fiber) "
        "and add it as a percentage column to a metrics CSV."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True, help="CSV with an 'image' column to augment.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: overwrite --csv in place).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs = find_pairs(cfg.data)

    fraction_by_filename: dict[str, float] = {}
    for i, pair in enumerate(pairs, start=1):
        mask = _read_gray(pair.mask_path)
        fraction = 100.0 * float((mask > 0).mean())
        fraction_by_filename[pair.image_path.name] = fraction
        print(f"[{i}/{len(pairs)}] {pair.image_path.name}: {fraction:.3f}%")

    df = pd.read_csv(args.csv)
    df[NEW_COLUMN] = df["image"].map(fraction_by_filename)

    anchor = next(
        (c for c in ("Median Otsu Coverage (%)", "Filter Coverage (%)") if c in df.columns),
        None,
    )
    if anchor:
        cols = [c for c in df.columns if c != NEW_COLUMN]
        insert_at = cols.index(anchor) + 1
        cols.insert(insert_at, NEW_COLUMN)
        df = df[cols]

    unmatched = df[NEW_COLUMN].isna().sum()
    if unmatched:
        print(f"Warning: {unmatched} rows in --csv had no matching ground-truth mask computed.")

    out_path = Path(args.out) if args.out else Path(args.csv)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
