# annotator_agreement.py
"""Inter-annotator agreement: how well do two sets of masks for the same images agree?

This is the label-quality ceiling. Have a second person (or the same person, weeks
later, blind to the first labels) re-annotate ~10 images, then score one set against the
other with the same metrics the model is scored with. If two humans only reach
dice ~0.75 on these fibers, a model at 0.68 is close to the ceiling and more modelling
buys little - better labels (or a tolerance-aware metric) would. If humans reach 0.95,
there is real headroom.

Masks are paired by filename (both folders must use the same names). Set A is treated as
the reference, so precision/recall are asymmetric; dice, tol_f1 and cldice are symmetric.

Run:
    python -m fiberseg.tools.annotator_agreement --masks-a Masks --masks-b Masks_relabel
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from ..dataset import _read_gray
from .evaluate_predictions import DEFAULT_TOLERANCE_PX, FIELDNAMES, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Score two mask folders against each other.")
    parser.add_argument("--masks-a", required=True, help="Reference annotation folder.")
    parser.add_argument("--masks-b", required=True, help="Second annotation folder.")
    parser.add_argument("--glob", default="*_mask.tif")
    parser.add_argument("--tolerance-px", type=float, default=DEFAULT_TOLERANCE_PX)
    parser.add_argument("--out", default=None, help="Default: <masks-b>/agreement.csv.")
    args = parser.parse_args()

    a_dir, b_dir = Path(args.masks_a), Path(args.masks_b)
    rows = []
    for b_path in sorted(b_dir.glob(args.glob)):
        a_path = a_dir / b_path.name
        if not a_path.exists():
            print(f"skip {b_path.name}: not in {a_dir}")
            continue
        a, b = _read_gray(a_path), _read_gray(b_path)
        if a.shape != b.shape:
            print(f"skip {b_path.name}: shape {b.shape} != {a.shape}")
            continue
        m = compute_metrics(b, a, tolerance_px=args.tolerance_px)
        rows.append({"image": b_path.name, "split": "", **m})
    if not rows:
        raise SystemExit("No mask pairs found.")

    out = Path(args.out) if args.out else b_dir / "agreement.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} images -> {out}")
    for k in ("dice", "precision", "recall", "tol_f1", "cldice"):
        vals = [r[k] for r in rows]
        print(f"  {k:10s} mean {statistics.fmean(vals):.4f}  "
              f"min {min(vals):.4f}  max {max(vals):.4f}")


if __name__ == "__main__":
    main()
