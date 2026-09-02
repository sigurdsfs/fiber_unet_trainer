# evaluate_predictions.py
"""Compute per-image binary segmentation metrics against ground truth and write a CSV.

Matches each prediction mask in `--pred-dir` (named `{image_stem}{suffix}`, the convention
used by `predict_tiles.py`/`predict_all.py`) back to its source image via the config's
`data.images_dir`/`data.mask_pattern` (same pairing logic as `dataset.find_pairs`), then
compares it against the corresponding ground-truth mask pixel-by-pixel.

Looks for each prediction under `--pred-dir/<train|validation|test>/` first, matching
`predict_all.py`'s split-subfolder layout, then falls back to a flat `--pred-dir` for
predictions written by `predict_tiles.py` (single image) or older flat `predict_all.py` runs.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from ..config import load_config
from ..dataset import SPLIT_DIRS, _read_gray, find_pairs

FIELDNAMES = [
    "image", "split",
    "accuracy", "precision", "recall", "specificity",
    "dice", "iou", "tversky", "f2",
    "tp", "fp", "fn", "tn",
]

# Kept separate from FIELDNAMES (not a compute_metrics() output) so callers
# that derive a "metric names" list from FIELDNAMES[2:] - predict_all.py and
# postprocess_masks.py, for their raw_*/post_*/delta_* column expansion -
# don't pick this up and try to look it up in a metrics dict that doesn't
# have it. It depends only on the ground-truth mask, not the prediction, so
# it never gets a raw/post/delta split anyway - just added once per row.
GT_FRACTION_COLUMN = "GT Foreground Fraction (%)"


def gt_foreground_fraction(mask: np.ndarray) -> float:
    """Percentage of `mask` that is foreground (`mask > 0`, this codebase's
    convention throughout)."""
    return 100.0 * float((mask > 0).mean())


def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Binary segmentation metrics for one image, using the same tp/fp/fn formulas as
    `lit_module._stats_from_counts` plus classic accuracy/specificity/confusion counts.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Prediction shape {pred_mask.shape} does not match ground truth shape "
            f"{gt_mask.shape}."
        )

    pred = pred_mask > 0
    targ = gt_mask > 0

    tp = float(np.sum(pred & targ))
    fp = float(np.sum(pred & ~targ))
    fn = float(np.sum(~pred & targ))
    tn = float(np.sum(~pred & ~targ))

    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "tversky": tp / (tp + alpha * fp + beta * fn + eps),
        "f2": (5 * tp) / (5 * tp + 4 * fn + fp + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-image segmentation metrics for a folder of prediction "
        "masks against the ground truth referenced by a config, and save them to a CSV."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir", required=True, help="Folder containing prediction masks.")
    parser.add_argument(
        "--suffix",
        default="_pred.tif",
        help="Filename suffix used for prediction masks, matching predict_all.py's "
        "--suffix (default: _pred.tif).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <pred-dir>/metrics.csv).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(args.pred_dir)

    pairs = find_pairs(cfg.data)

    rows = []
    missing = []

    for pair in pairs:
        # predict_all.py writes into a train/validation/test subfolder by split; older
        # runs (or predict_tiles.py on a single image) may have written flat instead.
        candidates = [
            pred_dir / SPLIT_DIRS[pair.split] / f"{pair.image_path.stem}{args.suffix}",
            pred_dir / f"{pair.image_path.stem}{args.suffix}",
        ]
        pred_path = next((p for p in candidates if p.exists()), None)
        if pred_path is None:
            missing.append(candidates[0])
            continue

        gt_mask = _read_gray(pair.mask_path)
        pred_mask = _read_gray(pred_path)

        metrics = compute_metrics(
            pred_mask,
            gt_mask,
            alpha=cfg.train.loss.tversky_alpha,
            beta=cfg.train.loss.tversky_beta,
        )
        row = {"image": pair.image_path.name, "split": pair.split}
        row[GT_FRACTION_COLUMN] = gt_foreground_fraction(gt_mask)
        row.update(metrics)
        rows.append(row)

    if missing:
        print(
            f"Warning: {len(missing)} prediction files were not found in {pred_dir} "
            f"(first missing: {missing[0].name})."
        )

    if not rows:
        raise FileNotFoundError(
            f"No matching prediction files found in {pred_dir} using suffix {args.suffix!r}."
        )

    output_fieldnames = FIELDNAMES[:2] + [GT_FRACTION_COLUMN] + FIELDNAMES[2:]
    out_path = Path(args.out) if args.out else pred_dir / "metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {k: statistics.fmean(row[k] for row in rows) for k in FIELDNAMES[2:]}
    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Mean metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in means.items()))


if __name__ == "__main__":
    main()
