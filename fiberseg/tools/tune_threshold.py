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

import numpy as np

from ..config import load_config
from ..dataset import _normalize_image, _read_gray, find_pairs
from ..predict_tiles import load_predictor, predict_prob


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    pairs = [p for p in find_pairs(cfg.data) if p.split == args.split]
    if not pairs:
        raise SystemExit(f"No images in split {args.split!r}.")

    thresholds = np.linspace(0.0, 1.0, args.steps + 2)[1:-1]
    tp = np.zeros_like(thresholds, dtype=np.float64)
    fp = np.zeros_like(thresholds, dtype=np.float64)
    fn = np.zeros_like(thresholds, dtype=np.float64)

    for i, pair in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] {pair.image_path.name}")
        img = _normalize_image(_read_gray(pair.image_path))
        prob = predict_prob(img, model, cfg, device)
        gt = _read_gray(pair.mask_path)
        t_, f_, n_ = _counts_at(prob, gt, thresholds)
        tp += t_
        fp += f_
        fn += n_

    eps = 1e-8
    a, b = cfg.train.loss.tversky_alpha, cfg.train.loss.tversky_beta
    metrics = {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "f2": (5 * tp) / (5 * tp + 4 * fn + fp + eps),
        "tversky": tp / (tp + a * fp + b * fn + eps),
    }
    score = metrics[args.metric]
    best = int(np.argmax(score))

    print("=" * 60)
    print(f"Tuned on split={args.split!r}, maximizing {args.metric!r} "
          f"(micro-averaged over {len(pairs)} images)")
    print(f"  default threshold 0.5 -> {args.metric}="
          f"{score[np.argmin(np.abs(thresholds - 0.5))]:.4f}")
    print(f"  BEST threshold {thresholds[best]:.3f} -> {args.metric}={score[best]:.4f}")
    print(f"  (dice={metrics['dice'][best]:.4f}, iou={metrics['iou'][best]:.4f}, "
          f"f2={metrics['f2'][best]:.4f})")
    print("=" * 60)
    print(f"Set `train.threshold: {thresholds[best]:.3f}` in your config to use it.")


if __name__ == "__main__":
    main()
