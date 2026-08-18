# predict_all.py
"""Batch tiled inference over every image/mask pair referenced by a config.

Loads a checkpoint once via `predict_tiles.load_predictor`, then runs
`predict_tiles.predict_mask` for each pair found via `dataset.find_pairs` (same
discovery/exclusion/split rule used for training), writing each predicted mask into a
`train`/`validation`/`test` subfolder of `--out-dir` matching that image's split - so
predictions can be reviewed or scored per split without hand-sorting files afterwards.

Also runs every raw prediction through `tools.postprocess_masks.postprocess_mask` (classical,
non-learned morphological cleanup - closing, opening, small-object removal, small-hole
filling), writing the processed mask into a parallel `postprocessed/train`/`validation`/`test`
subfolder of `--out-dir` alongside the raw one. Both are scored against ground truth
(`tools.evaluate_predictions.compute_metrics`) into a single `--out-dir/metrics.csv`: each row
has `raw_*`, `post_*`, and `delta_*` (post minus raw) columns per metric, so
`evaluate_predictions.py` never needs a separate run for either version.

`--tune-threshold` optionally re-tunes `train.threshold` first, sweeping it on a single
split (`--tune-split`, default `val`) via `tools.tune_threshold.find_best_threshold`, then
applies that one threshold uniformly across every image regardless of split - also saving
the sweep as a precision-recall curve (`--out-dir/threshold_pr_curve.png`, AUC-PR in the
title; precision-recall rather than ROC since fiber masks are sparse/background-dominated).
Either way, `--out-dir/threshold_info.txt` records whether tuning was on, the
split/metric/step count used, and the threshold actually applied.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime
from pathlib import Path

from .config import load_config
from .dataset import SPLIT_DIRS, _normalize_image, _read_gray, find_pairs
from .predict_tiles import load_predictor, predict_mask, save_mask
from .tools.evaluate_predictions import FIELDNAMES, compute_metrics
from .tools.postprocess_masks import postprocess_mask
from .tools.tune_threshold import find_best_threshold, plot_pr_curve

# Per-metric columns in metrics.csv: metric names come from evaluate_predictions.FIELDNAMES,
# minus the leading "image"/"split" columns.
METRIC_NAMES = FIELDNAMES[2:]
COMBINED_FIELDNAMES = (
    ["image", "split"]
    + [f"raw_{m}" for m in METRIC_NAMES]
    + [f"post_{m}" for m in METRIC_NAMES]
    + [f"delta_{m}" for m in METRIC_NAMES]
)


def main():
    """CLI entry point: predict a mask for every image/mask pair in the config, sorted
    into train/validation/test subfolders of `--out-dir` by split."""
    parser = argparse.ArgumentParser(
        description="Run tiled prediction on every image/mask pair referenced by a config, "
        "writing outputs into train/validation/test subfolders of --out-dir by split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--suffix",
        default="_pred.tif",
        help="Filename suffix appended to each image stem for the output mask (default: _pred.tif).",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Re-tune train.threshold on --tune-split before predicting, then use that "
        "single threshold for every image regardless of split.",
    )
    parser.add_argument(
        "--tune-split",
        default="val",
        choices=["train", "val", "test"],
        help="Split to tune train.threshold on when --tune-threshold is set (default: val; "
        "never tune on test for reporting).",
    )
    parser.add_argument(
        "--tune-metric",
        default="dice",
        choices=["dice", "iou", "f2", "tversky"],
        help="Metric to maximize when --tune-threshold is set (default: dice).",
    )
    parser.add_argument(
        "--tune-steps",
        type=int,
        default=99,
        help="Thresholds tried in (0,1) when --tune-threshold is set (default: 99).",
    )
    parser.add_argument(
        "--closing-radius", type=int, default=1,
        help="Post-processing: morphological closing disk radius; 0 disables (default: 1). "
        "Bridges small gaps in broken fibers.",
    )
    parser.add_argument(
        "--opening-radius", type=int, default=0,
        help="Post-processing: morphological opening disk radius; 0 disables (default: 0). "
        "Smooths jagged edges, but can erode away thin fibers if set too large.",
    )
    parser.add_argument(
        "--min-object-size", type=int, default=64,
        help="Post-processing: remove connected components smaller than this many pixels; "
        "0 disables (default: 64).",
    )
    parser.add_argument(
        "--max-hole-size", type=int, default=64,
        help="Post-processing: fill interior holes smaller than this many pixels; 0 disables "
        "(default: 64).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tune_stats = None
    pr_auc = None
    if args.tune_threshold:
        best_threshold, tune_stats, tune_thresholds, tune_metrics = find_best_threshold(
            cfg, model, device,
            split=args.tune_split, metric=args.tune_metric, steps=args.tune_steps,
        )
        print(
            f"Tuned train.threshold on the {args.tune_split} split: {best_threshold:.3f} "
            f"({args.tune_metric}={tune_stats[args.tune_metric]:.4f}); using it for all images."
        )
        cfg.train.threshold = best_threshold

        pr_curve_path = out_dir / "threshold_pr_curve.png"
        pr_auc = plot_pr_curve(
            tune_thresholds, tune_metrics, best_threshold, pr_curve_path,
            title=f"Precision-recall curve, split={args.tune_split!r}",
        )
        print(f"Wrote precision-recall curve (AUC-PR={pr_auc:.4f}) to {pr_curve_path}")

    pairs = find_pairs(cfg.data)

    split_out_dirs = {split: out_dir / name for split, name in SPLIT_DIRS.items()}
    postprocessed_out_dirs = {
        split: out_dir / "postprocessed" / name for split, name in SPLIT_DIRS.items()
    }
    for d in list(split_out_dirs.values()) + list(postprocessed_out_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, pair in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] Predicting {pair.image_path.name} ({pair.split}) ...")

        img = _normalize_image(_read_gray(pair.image_path))
        mask = predict_mask(img, model, cfg, device)

        out_path = split_out_dirs[pair.split] / f"{pair.image_path.stem}{args.suffix}"
        save_mask(mask, out_path)

        processed = postprocess_mask(
            mask,
            closing_radius=args.closing_radius,
            opening_radius=args.opening_radius,
            min_object_size=args.min_object_size,
            max_hole_size=args.max_hole_size,
        )
        processed_out_path = (
            postprocessed_out_dirs[pair.split] / f"{pair.image_path.stem}{args.suffix}"
        )
        save_mask(processed, processed_out_path)

        gt_mask = _read_gray(pair.mask_path)
        raw_metrics = compute_metrics(
            mask, gt_mask,
            alpha=cfg.train.loss.tversky_alpha, beta=cfg.train.loss.tversky_beta,
        )
        post_metrics = compute_metrics(
            processed, gt_mask,
            alpha=cfg.train.loss.tversky_alpha, beta=cfg.train.loss.tversky_beta,
        )

        row = {"image": pair.image_path.name, "split": pair.split}
        for m in METRIC_NAMES:
            row[f"raw_{m}"] = raw_metrics[m]
            row[f"post_{m}"] = post_metrics[m]
            row[f"delta_{m}"] = post_metrics[m] - raw_metrics[m]
        rows.append(row)

    counts = {split: sum(1 for p in pairs if p.split == split) for split in SPLIT_DIRS}
    print(
        f"Done. Wrote {len(pairs)} predictions to {out_dir} "
        f"(train={counts['train']}, validation={counts['val']}, test={counts['test']})"
    )

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    raw_means = {m: statistics.fmean(row[f"raw_{m}"] for row in rows) for m in METRIC_NAMES}
    post_means = {m: statistics.fmean(row[f"post_{m}"] for row in rows) for m in METRIC_NAMES}
    delta_means = {m: post_means[m] - raw_means[m] for m in METRIC_NAMES}
    print(f"Wrote metrics for {len(rows)} images to {metrics_path}")
    print(
        "Raw mean metrics:            "
        + ", ".join(f"{k}={v:.4f}" for k, v in raw_means.items())
    )
    print(
        "Post-processed mean metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in post_means.items())
    )
    print(
        "Mean delta (post - raw):     "
        + ", ".join(f"{k}={v:+.4f}" for k, v in delta_means.items())
    )

    info_path = out_dir / "threshold_info.txt"
    info_lines = [
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Config: {args.config}",
        f"Checkpoint: {args.checkpoint}",
        "",
        f"Threshold tuning: {'ON' if args.tune_threshold else 'OFF'}",
    ]
    if args.tune_threshold:
        info_lines += [
            f"  Tuned on split: {args.tune_split}",
            f"  Metric maximized: {args.tune_metric}",
            f"  Thresholds swept: {args.tune_steps}",
            "  Stats at best threshold: "
            + ", ".join(f"{k}={v:.4f}" for k, v in tune_stats.items()),
            f"  Precision-recall AUC: {pr_auc:.4f} (see threshold_pr_curve.png)",
        ]
    else:
        info_lines.append("  (using train.threshold from the config, unchanged)")
    info_lines += [
        "",
        f"Threshold used for all predictions: {cfg.train.threshold:.3f}",
        f"Images processed: {len(pairs)} "
        f"(train={counts['train']}, validation={counts['val']}, test={counts['test']})",
        "",
        "Post-processing (classical, non-learned): closing_radius="
        f"{args.closing_radius}, opening_radius={args.opening_radius}, "
        f"min_object_size={args.min_object_size}, max_hole_size={args.max_hole_size}",
        "  Mean delta (post - raw): "
        + ", ".join(f"{k}={v:+.4f}" for k, v in delta_means.items()),
    ]
    info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")
    print(f"Wrote threshold info to {info_path}")


if __name__ == "__main__":
    main()
