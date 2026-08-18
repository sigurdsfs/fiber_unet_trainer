# postprocess_masks.py
"""Classical (non-learned) mask post-processing, applied to predictions and re-scored.

`postprocess_mask` is a small morphology pipeline - closing, then opening, then
small-object removal, then small-hole filling - no neural network involved, just
skimage.morphology. Order matters: closing runs first so it can bridge genuinely-broken
single fibers into one component *before* the small-object filter would otherwise discard
each fragment as noise.

The CLI applies it to every prediction written by `predict_all.py` (or `predict_tiles.py`),
writing processed masks into `--out-dir` (default `<pred-dir>/postprocessed`) using the same
`train`/`validation`/`test` subfolder layout, then recomputes the same metrics as
`evaluate_predictions.py` against ground truth for the processed masks and writes
`--out-dir/metrics.csv`. If `--pred-dir/metrics.csv` already exists (predict_all.py writes
one), also prints a before/after mean-metric comparison.

Run:
    python -m fiberseg.tools.postprocess_masks --config <cfg> --pred-dir <predict_all out-dir>
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from skimage.morphology import (
    closing,
    disk,
    opening,
    remove_small_holes,
    remove_small_objects,
)

from ..config import load_config
from ..dataset import SPLIT_DIRS, _read_gray, find_pairs
from ..predict_tiles import save_mask
from .evaluate_predictions import FIELDNAMES, compute_metrics


def postprocess_mask(
    mask: np.ndarray,
    *,
    closing_radius: int = 1,
    opening_radius: int = 0,
    min_object_size: int = 64,
    max_hole_size: int = 64,
) -> np.ndarray:
    """Classical cleanup of a binary fiber mask. `mask` and the return value are both
    0/255 uint8, matching `predict_tiles.save_mask`'s convention. Any step is skipped by
    passing 0 for its parameter.
    """
    binary = mask > 0

    if closing_radius > 0:
        binary = closing(binary, footprint=disk(closing_radius))
    if opening_radius > 0:
        binary = opening(binary, footprint=disk(opening_radius))
    if min_object_size > 0:
        binary = remove_small_objects(binary, max_size=min_object_size)
    if max_hole_size > 0:
        binary = remove_small_holes(binary, max_size=max_hole_size)

    return binary.astype(np.uint8) * 255


def _find_prediction(pred_dir: Path, split: str, stem: str, suffix: str) -> Path | None:
    """Mirror evaluate_predictions.py's lookup: split subfolder first, flat as fallback."""
    candidates = [
        pred_dir / SPLIT_DIRS[split] / f"{stem}{suffix}",
        pred_dir / f"{stem}{suffix}",
    ]
    return next((p for p in candidates if p.exists()), None)


def main():
    parser = argparse.ArgumentParser(
        description="Apply classical mask post-processing to predictions from "
        "predict_all.py/predict_tiles.py and recompute metrics against ground truth."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pred-dir", required=True,
        help="Folder of raw predictions, e.g. predict_all.py's --out-dir.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write processed masks + metrics.csv (default: <pred-dir>/postprocessed).",
    )
    parser.add_argument(
        "--suffix", default="_pred.tif",
        help="Filename suffix used for prediction masks (default: _pred.tif).",
    )
    parser.add_argument(
        "--closing-radius", type=int, default=1,
        help="Morphological closing disk radius; 0 disables (default: 1). Bridges small "
        "gaps in broken fibers.",
    )
    parser.add_argument(
        "--opening-radius", type=int, default=0,
        help="Morphological opening disk radius; 0 disables (default: 0). Smooths jagged "
        "edges, but can erode away thin fibers if set too large.",
    )
    parser.add_argument(
        "--min-object-size", type=int, default=64,
        help="Remove connected components smaller than this many pixels; 0 disables "
        "(default: 64).",
    )
    parser.add_argument(
        "--max-hole-size", type=int, default=64,
        help="Fill interior holes smaller than this many pixels; 0 disables (default: 64).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs = find_pairs(cfg.data)

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir / "postprocessed"
    split_out_dirs = {split: out_dir / name for split, name in SPLIT_DIRS.items()}
    for d in split_out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for i, pair in enumerate(pairs, start=1):
        pred_path = _find_prediction(pred_dir, pair.split, pair.image_path.stem, args.suffix)
        if pred_path is None:
            missing.append(pair.image_path.name)
            continue

        print(f"[{i}/{len(pairs)}] Post-processing {pair.image_path.name} ({pair.split}) ...")
        raw_mask = _read_gray(pred_path)
        processed = postprocess_mask(
            raw_mask,
            closing_radius=args.closing_radius,
            opening_radius=args.opening_radius,
            min_object_size=args.min_object_size,
            max_hole_size=args.max_hole_size,
        )

        out_path = split_out_dirs[pair.split] / f"{pair.image_path.stem}{args.suffix}"
        save_mask(processed, out_path)

        gt_mask = _read_gray(pair.mask_path)
        metrics = compute_metrics(
            processed, gt_mask,
            alpha=cfg.train.loss.tversky_alpha, beta=cfg.train.loss.tversky_beta,
        )
        rows.append({"image": pair.image_path.name, "split": pair.split, **metrics})

    if missing:
        print(
            f"Warning: {len(missing)} predictions were not found in {pred_dir} "
            f"(first missing: {missing[0]})."
        )
    if not rows:
        raise FileNotFoundError(
            f"No matching prediction files found in {pred_dir} using suffix {args.suffix!r}."
        )

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    means = {k: statistics.fmean(row[k] for row in rows) for k in FIELDNAMES[2:]}
    print(f"Wrote metrics for {len(rows)} processed masks to {metrics_path}")
    print("Post-processed mean metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in means.items()))

    raw_metrics_path = pred_dir / "metrics.csv"
    if raw_metrics_path.exists():
        with open(raw_metrics_path, newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        processed_images = {r["image"] for r in rows}
        raw_rows = [r for r in raw_rows if r["image"] in processed_images]
        if raw_rows:
            raw_means = {
                k: statistics.fmean(float(row[k]) for row in raw_rows) for k in FIELDNAMES[2:]
            }
            print(
                "Raw (pre-processing) mean metrics, same images: "
                + ", ".join(f"{k}={v:.4f}" for k, v in raw_means.items())
            )
            print(
                "Delta (post - raw): "
                + ", ".join(f"{k}={means[k] - raw_means[k]:+.4f}" for k in FIELDNAMES[2:])
            )
    else:
        print(
            f"(No {raw_metrics_path} found - run predict_all.py or evaluate_predictions.py "
            "first for a before/after comparison.)"
        )


if __name__ == "__main__":
    main()
