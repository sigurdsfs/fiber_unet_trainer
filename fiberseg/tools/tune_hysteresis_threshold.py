# tune_hysteresis_threshold.py
"""Find the hysteresis `low`/`high` pair that maximizes a metric on the validation split.

Same idea as `tune_threshold.py` - sweep on validation probability maps built from the
same tiled inference the real predictor uses, no retrain - but for
`cfg.inference.hysteresis_low`/`hysteresis_high` (`inference.threshold_mode: "hysteresis"`,
see `tools.fiber_gap_repair.hysteresis_threshold_mask`) instead of the single fixed
`train.threshold`.

Unlike a plain threshold, hysteresis isn't a pointwise cut - a pixel's inclusion depends
on connectivity to a confident seed region, which changes with both `low` and `high`. That
rules out `tune_threshold.py`'s O(1)-per-threshold sort/searchsorted trick, so this instead
grid-searches `(low, high)` pairs directly: each image's probability map is still computed
once (the expensive, GPU part), but every valid `low < high` combination in the grid is
evaluated with an actual `apply_hysteresis_threshold` call (cheap, CPU-only, but not free -
budget for `n_images * n_valid_pairs` calls at roughly 50ms each per megapixel; keep
`--low-steps`/`--high-steps` modest, or narrow `--low-max`/`--high-min` around a rough
guess, for a large validation split).

Run:
    python -m fiberseg.tools.tune_hysteresis_threshold --config <cfg> --checkpoint <best.ckpt>

Apply the printed values by setting `inference.threshold_mode: hysteresis` plus
`inference.hysteresis_low`/`inference.hysteresis_high` in your config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import load_config
from ..dataset import _normalize_image, _read_gray, find_pairs
from ..predict_tiles import load_predictor, predict_prob
from .fiber_gap_repair import hysteresis_threshold_mask

METRIC_CHOICES = ["dice", "iou", "f2", "tversky"]


def sweep_hysteresis_thresholds(
    cfg, model, device, *,
    split: str = "val",
    low_steps: int = 20,
    high_steps: int = 20,
    low_min: float = 0.0,
    low_max: float = 1.0,
    high_min: float = 0.0,
    high_max: float = 1.0,
    verbose: bool = True,
):
    """Grid-search `(low, high)` on `split` and return
    `(lows, highs, metrics, valid, n_images)`.

    `lows`/`highs` are 1D candidate arrays (`np.linspace(*_min, *_max, *_steps)`).
    `metrics` maps each metric name to a `(low_steps, high_steps)` array, micro-averaged
    over all images (accumulate tp/fp/fn per cell, then one ratio at the end - same
    pattern as `lit_module`'s epoch-level metrics, not per-image-then-averaged). Cells
    where `low >= high` are invalid and set to `-inf` in every metric array so they never
    win an `argmax`; `valid` is the boolean mask marking which cells were actually
    evaluated. Factored out of `main()` so other entry points can reuse the sweep.
    """
    pairs = [p for p in find_pairs(cfg.data) if p.split == split]
    if not pairs:
        raise SystemExit(f"No images in split {split!r}.")

    lows = np.linspace(low_min, low_max, low_steps)
    highs = np.linspace(high_min, high_max, high_steps)
    valid = lows[:, None] < highs[None, :]

    tp = np.zeros((low_steps, high_steps), dtype=np.float64)
    fp = np.zeros_like(tp)
    fn = np.zeros_like(tp)

    n_valid = int(valid.sum())
    for i, pair in enumerate(pairs, start=1):
        if verbose:
            print(f"[{i}/{len(pairs)}] {pair.image_path.name} ({n_valid} (low, high) pairs)")
        img = _normalize_image(_read_gray(pair.image_path))
        prob = predict_prob(img, model, cfg, device)
        fiber = _read_gray(pair.mask_path) > 0

        for li, low in enumerate(lows):
            for hi, high in enumerate(highs):
                if not valid[li, hi]:
                    continue
                pred = hysteresis_threshold_mask(prob, low, high)
                tp[li, hi] += np.sum(pred & fiber)
                fp[li, hi] += np.sum(pred & ~fiber)
                fn[li, hi] += np.sum(~pred & fiber)

    eps = 1e-8
    a, b = cfg.train.loss.tversky_alpha, cfg.train.loss.tversky_beta
    raw_metrics = {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "f2": (5 * tp) / (5 * tp + 4 * fn + fp + eps),
        "tversky": tp / (tp + a * fp + b * fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }
    metrics = {k: np.where(valid, v, -np.inf) for k, v in raw_metrics.items()}
    return lows, highs, metrics, valid, len(pairs)


def plot_hysteresis_heatmap(
    lows, highs, metric_grid, valid, best_low, best_high, out_path, *, title=None
):
    """Heatmap of `metric_grid` over the `(low, high)` grid, with invalid (`low >= high`)
    cells left blank and the chosen best pair marked. Saved to `out_path`; mirrors
    `tune_threshold.plot_pr_curve`'s role of visualizing the full sweep, not just the
    single best value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = np.ma.masked_where(~valid, metric_grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.9")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        display.T, origin="lower", cmap=cmap, aspect="auto",
        extent=[lows.min(), lows.max(), highs.min(), highs.max()],
    )
    ax.scatter([best_low], [best_high], color="#B97A2A", edgecolor="white", zorder=5,
               label=f"low={best_low:.3f}, high={best_high:.3f}")
    ax.set_xlabel("low")
    ax.set_ylabel("high")
    ax.set_title(title or "Hysteresis threshold sweep")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_best_hysteresis_threshold(
    cfg, model, device, *,
    split: str = "val",
    metric: str = "dice",
    low_steps: int = 20,
    high_steps: int = 20,
    low_min: float = 0.0,
    low_max: float = 1.0,
    high_min: float = 0.0,
    high_max: float = 1.0,
    verbose: bool = True,
):
    """Grid-search `(low, high)` on `split` and return
    `(best_low, best_high, stats_at_best, lows, highs, metrics, valid)` - the last three
    are the raw sweep arrays from `sweep_hysteresis_thresholds`, handed back so callers can
    plot the sweep (see `plot_hysteresis_heatmap`) without re-running it.

    Defaults to the validation split, matching this module's own CLI default and
    `tune_threshold.find_best_threshold` - avoid `split="test"` if the tuned values will
    then be applied across all splits, since tuning against test data would contaminate
    its evaluation.
    """
    lows, highs, metrics, valid, _ = sweep_hysteresis_thresholds(
        cfg, model, device, split=split,
        low_steps=low_steps, high_steps=high_steps,
        low_min=low_min, low_max=low_max, high_min=high_min, high_max=high_max,
        verbose=verbose,
    )
    score = metrics[metric]
    li, hi = np.unravel_index(int(np.argmax(score)), score.shape)
    stats = {k: float(v[li, hi]) for k, v in metrics.items()}
    return float(lows[li]), float(highs[hi]), stats, lows, highs, metrics, valid


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search the hysteresis low/high thresholds on the validation "
        "split and report the pair that maximizes the chosen metric."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", default="val", choices=["val", "test", "train"],
        help="Which split to tune on (default: val; never tune on test for reporting).",
    )
    parser.add_argument(
        "--metric", default="dice", choices=METRIC_CHOICES,
        help="Metric to maximize (default: dice).",
    )
    parser.add_argument(
        "--low-steps", type=int, default=20, help="Candidate `low` values (default: 20)."
    )
    parser.add_argument(
        "--high-steps", type=int, default=20, help="Candidate `high` values (default: 20)."
    )
    parser.add_argument(
        "--low-min", type=float, default=0.0, help="Smallest `low` to try (default: 0.0)."
    )
    parser.add_argument(
        "--low-max", type=float, default=1.0, help="Largest `low` to try (default: 1.0)."
    )
    parser.add_argument(
        "--high-min", type=float, default=0.0, help="Smallest `high` to try (default: 0.0)."
    )
    parser.add_argument(
        "--high-max", type=float, default=1.0, help="Largest `high` to try (default: 1.0)."
    )
    parser.add_argument(
        "--plot", default=None,
        help="Optional path to save a heatmap of the metric over the (low, high) grid, "
        "e.g. hysteresis_sweep.png.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    lows, highs, metrics, valid, n_images = sweep_hysteresis_thresholds(
        cfg, model, device, split=args.split,
        low_steps=args.low_steps, high_steps=args.high_steps,
        low_min=args.low_min, low_max=args.low_max,
        high_min=args.high_min, high_max=args.high_max,
    )
    score = metrics[args.metric]
    li, hi = np.unravel_index(int(np.argmax(score)), score.shape)
    best_low, best_high = float(lows[li]), float(highs[hi])

    default_low = cfg.inference.hysteresis_low
    default_high = cfg.inference.hysteresis_high
    dli = int(np.argmin(np.abs(lows - default_low)))
    dhi = int(np.argmin(np.abs(highs - default_high)))

    print("=" * 60)
    print(f"Tuned on split={args.split!r}, maximizing {args.metric!r} "
          f"(micro-averaged over {n_images} images)")
    if valid[dli, dhi]:
        print(f"  config default low={default_low:.3f}, high={default_high:.3f} -> "
              f"{args.metric}={score[dli, dhi]:.4f}")
    print(f"  BEST low={best_low:.3f}, high={best_high:.3f} -> {args.metric}={score[li, hi]:.4f}")
    print(f"  (dice={metrics['dice'][li, hi]:.4f}, iou={metrics['iou'][li, hi]:.4f}, "
          f"f2={metrics['f2'][li, hi]:.4f})")
    print("=" * 60)
    print(f"Set `inference.threshold_mode: hysteresis`, `inference.hysteresis_low: "
          f"{best_low:.3f}`, `inference.hysteresis_high: {best_high:.3f}` in your config.")

    if args.plot:
        plot_hysteresis_heatmap(
            lows, highs, score, valid, best_low, best_high, args.plot,
            title=f"Hysteresis {args.metric} sweep, split={args.split!r}",
        )
        print(f"Wrote hysteresis sweep heatmap to {args.plot}")


if __name__ == "__main__":
    main()
