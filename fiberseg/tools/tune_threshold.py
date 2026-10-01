# tune_threshold.py
"""Find the decision threshold that maximizes a metric on the validation split.

The training loss is deliberately recall-weighted (tversky_beta > tversky_alpha),
so the model's sigmoid outputs are biased and the default 0.5 cut is almost never
the dice/iou-optimal operating point. This sweeps thresholds over the validation
images' full-resolution probability maps (built with the same tiled inference the
real predictor uses) and reports the best one - a seconds-to-minutes, no-retrain
gain of typically 1-3 dice points.

Run:
    python -m fiberseg.tools.tune_threshold --config <cfg> --checkpoint <best.ckpt>

Apply the printed threshold by setting `train.threshold` in your config before
running predict_all / evaluate_predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import load_config
from ..dataset import _normalize_image, _read_gray, find_pairs
from ..predict_tiles import load_predictor, predict_prob


def _tuning_grid(steps: int) -> np.ndarray:
    """The `steps` evenly spaced candidate thresholds in (0, 1) that tuning picks from."""
    return np.linspace(0.0, 1.0, steps + 2)[1:-1]


def _threshold_grid(steps: int) -> np.ndarray:
    """Tuning grid plus logit-spaced tails reaching ~1e-7 from 0 and 1.

    The tails exist for the precision-recall curve only. A recall-weighted model pushes
    most fiber pixels to probabilities > 0.99, so a linear grid stopping at 0.99 never
    reaches the low-recall end of the curve and the integrated area came out far too
    small (e.g. AUC-PR 0.27 next to a best dice of 0.68, which is impossible for a real
    curve). Tuning still only picks from `_tuning_grid` - see `_best_index`.
    """
    tails = 1.0 / (1.0 + np.exp(-np.linspace(-16.0, 16.0, 321)))
    return np.unique(np.concatenate([_tuning_grid(steps), tails]))


def _best_index(thresholds: np.ndarray, score: np.ndarray, steps: int) -> int:
    """Index into `thresholds` of the best-scoring threshold on the tuning grid."""
    candidates = np.flatnonzero(np.isin(thresholds, _tuning_grid(steps)))
    return int(candidates[np.argmax(score[candidates])])


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Step-wise average precision, AP = sum_n (R_n - R_{n+1}) * P_n over thresholds in
    increasing order, with the curve closed at recall 0 by the precision of the highest
    threshold (the sklearn `average_precision_score` convention). Step-wise rather than
    trapezoidal, which overestimates area between sparse PR points."""
    order = np.argsort(recall)
    r = np.concatenate([[0.0], np.asarray(recall)[order]])
    p = np.asarray(precision)[order]
    return float(np.sum(np.diff(r) * p))


def _counts_at(prob: np.ndarray, gt: np.ndarray, thresholds: np.ndarray):
    """tp/fp/fn for every threshold at once, accumulated over one image."""
    fiber = gt > 0
    pos = fiber.sum()
    # For each threshold t, pred = prob > t. Sort probabilities of fiber vs
    # background pixels once and count via searchsorted for all thresholds.
    fiber_probs = np.sort(prob[fiber])
    bg_probs = np.sort(prob[~fiber])
    # pixels predicted positive above threshold t:
    tp = pos - np.searchsorted(fiber_probs, thresholds, side="right")
    fp = bg_probs.size - np.searchsorted(bg_probs, thresholds, side="right")
    fn = pos - tp
    return tp.astype(np.float64), fp.astype(np.float64), fn.astype(np.float64)


def metrics_from_counts(tp, fp, fn, alpha: float, beta: float) -> dict:
    """dice/iou/f2/tversky/precision/recall from (arrays of) pooled tp/fp/fn counts."""
    eps = 1e-8
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "f2": (5 * tp) / (5 * tp + 4 * fn + fp + eps),
        "tversky": tp / (tp + alpha * fp + beta * fn + eps),
        # Not offered as a --metric choice (dice/iou/f2/tversky are the tuning targets),
        # but needed to plot/score the precision-recall curve.
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def sweep_thresholds(cfg, model, device, *, split="val", steps=99, verbose=True,
                     ridge=False):
    """Sweep `train.threshold` candidates on `split` and return (thresholds, metrics, n_images).

    `metrics` maps each metric name to an array aligned with `thresholds`. Factored out of
    `main()` so other entry points (e.g. `predict_all.py --tune-threshold`) can reuse the
    sweep without going through the CLI.

    With `ridge=True` the sweep runs over the Hessian ridge response
    (`tools.fiber_gap_repair.enhance_ridges`, configured by `cfg.inference.ridge_*`)
    instead of the raw probability map, which is what `inference.threshold_mode: "ridge"`
    thresholds at prediction time - so the value it returns calibrates
    `inference.ridge_threshold` rather than `train.threshold`.
    """
    pairs = [p for p in find_pairs(cfg.data) if p.split == split]
    if not pairs:
        raise SystemExit(f"No images in split {split!r}.")

    thresholds = _threshold_grid(steps)
    tp = np.zeros_like(thresholds, dtype=np.float64)
    fp = np.zeros_like(thresholds, dtype=np.float64)
    fn = np.zeros_like(thresholds, dtype=np.float64)

    for i, pair in enumerate(pairs, start=1):
        if verbose:
            print(f"[{i}/{len(pairs)}] {pair.image_path.name}")
        img = _normalize_image(_read_gray(pair.image_path))
        prob = predict_prob(img, model, cfg, device)
        if ridge:
            # Local import: fiber_gap_repair pulls save_mask from predict_tiles, and
            # importing it at module load time would create a cycle.
            from .fiber_gap_repair import enhance_ridges

            prob = enhance_ridges(
                prob,
                method=cfg.inference.ridge_method,
                sigmas=tuple(cfg.inference.ridge_sigmas),
                device=cfg.inference.ridge_device,
            )
        gt = _read_gray(pair.mask_path)
        t_, f_, n_ = _counts_at(prob, gt, thresholds)
        tp += t_
        fp += f_
        fn += n_

    metrics = metrics_from_counts(
        tp, fp, fn, cfg.train.loss.tversky_alpha, cfg.train.loss.tversky_beta
    )
    return thresholds, metrics, len(pairs)


def plot_pr_curve(thresholds, metrics, best_threshold, out_path, *, title=None):
    """Plot the precision-recall curve traced out by `sweep_thresholds` and save it to
    `out_path`. Returns the area under the curve as average precision (see
    `average_precision`).

    Precision-recall, not ROC, because fiber masks are sparse (mostly-background pixels):
    ROC-AUC stays misleadingly high under that imbalance since it's dominated by the huge
    true-negative count, while PR-AUC actually reflects foreground detection quality - the
    same reasoning behind this project's use of dice/iou/tversky over raw accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision = np.asarray(metrics["precision"])
    recall = np.asarray(metrics["recall"])

    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    auc = average_precision(precision, recall)

    best_idx = int(np.argmin(np.abs(thresholds - best_threshold)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall_sorted, precision_sorted, where="post", color="#2E7A74", linewidth=2)
    ax.fill_between(recall_sorted, precision_sorted, step="post", alpha=0.15, color="#2E7A74")
    ax.scatter(
        [recall[best_idx]], [precision[best_idx]],
        color="#B97A2A", zorder=5,
        label=f"threshold={best_threshold:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Precision-recall curve (AUC-PR={auc:.4f})")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return auc


def find_best_threshold(
    cfg, model, device, *, split="val", metric="dice", steps=99, verbose=True,
    ridge=False,
):
    """Sweep `train.threshold` on `split` and return
    `(best_threshold, stats_at_best, thresholds, metrics)` - the last two are the raw
    sweep arrays from `sweep_thresholds`, handed back so callers (e.g. `predict_all.py
    --tune-threshold`) can plot the sweep (see `plot_pr_curve`) without re-running it.

    Defaults to the validation split, matching this module's own CLI default: callers
    that then apply the tuned threshold across all splits (e.g. `predict_all.py
    --tune-threshold`, itself defaulting to `--tune-split val`) should avoid `split="test"`,
    since tuning against test data would contaminate its evaluation.
    """
    thresholds, metrics, _ = sweep_thresholds(
        cfg, model, device, split=split, steps=steps, verbose=verbose, ridge=ridge
    )
    best = _best_index(thresholds, metrics[metric], steps)
    stats = {k: float(v[best]) for k, v in metrics.items()}
    return float(thresholds[best]), stats, thresholds, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Sweep the decision threshold on the validation split and report "
        "the value that maximizes the chosen metric."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Which split to tune on (default: val; never tune on test for reporting).",
    )
    parser.add_argument(
        "--metric",
        default="dice",
        choices=["dice", "iou", "f2", "tversky"],
        help="Metric to maximize (default: dice).",
    )
    parser.add_argument("--steps", type=int, default=99, help="Thresholds tried in (0,1).")
    parser.add_argument(
        "--ridge",
        action="store_true",
        help="Sweep over the Hessian ridge response (enhance_ridges, configured by "
        "inference.ridge_*) rather than the raw probability map - i.e. calibrate "
        "inference.ridge_threshold for inference.threshold_mode: 'ridge'.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional path to save a precision-recall curve (AUC-PR) over the sweep, "
        "e.g. pr_curve.png.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    thresholds, metrics, n_images = sweep_thresholds(
        cfg, model, device, split=args.split, steps=args.steps, ridge=args.ridge
    )
    score = metrics[args.metric]
    best = _best_index(thresholds, score, args.steps)

    target_key = "inference.ridge_threshold" if args.ridge else "train.threshold"
    swept_over = (
        f"the {cfg.inference.ridge_method} ridge response" if args.ridge
        else "the probability map"
    )

    print("=" * 60)
    print(f"Tuned on split={args.split!r}, maximizing {args.metric!r} "
          f"(micro-averaged over {n_images} images, over {swept_over})")
    print(f"  default threshold 0.5 -> {args.metric}="
          f"{score[np.argmin(np.abs(thresholds - 0.5))]:.4f}")
    print(f"  BEST threshold {thresholds[best]:.3f} -> {args.metric}={score[best]:.4f}")
    print(f"  (dice={metrics['dice'][best]:.4f}, iou={metrics['iou'][best]:.4f}, "
          f"f2={metrics['f2'][best]:.4f})")
    print("=" * 60)
    print(f"Set `{target_key}: {thresholds[best]:.3f}` in your config to use it.")

    if args.plot:
        auc = plot_pr_curve(
            thresholds, metrics, thresholds[best], args.plot,
            title=f"Precision-recall curve, split={args.split!r}",
        )
        print(f"Wrote precision-recall curve (AUC-PR={auc:.4f}) to {args.plot}")


if __name__ == "__main__":
    main()
