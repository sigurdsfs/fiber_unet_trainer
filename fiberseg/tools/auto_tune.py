# auto_tune.py
"""Post-training threshold calibration: tune on validation, score on test, save in the checkpoint.

Runs automatically at the end of every training run (`train.auto_tune_threshold`,
default on) and can be run by hand to backfill an existing checkpoint.

For each mode in `train.auto_tune_modes` the decision rule is tuned on the VALIDATION
split to maximize `train.auto_tune_metric` (micro-averaged, full-image tiled inference):
    fixed      -> train.threshold                        (probability-map sweep)
    hysteresis -> inference.hysteresis_low/high           (exact grid, hysteresis_counts)
    ridge      -> inference.ridge_threshold              (sweep of the enhance_ridges response)
Every image's probability map is computed once and shared by all modes. The TEST split
is then scored per image for each mode at its tuned values (plus `fixed-untuned`, the
untuned config threshold, as a reference), using evaluate_predictions.compute_metrics -
so each `test_metrics_<mode>.csv` feeds straight into `tools.compare_runs`.

Outputs (in `out_dir`, and as MLflow artifacts under `threshold_tuning/` when a run id is
given): `tuned_thresholds.json`, `test_metrics_<mode>.csv`. The tuned values are also
written into the checkpoint under `tuned_thresholds.TUNED_KEY`, which
`predict_tiles.load_predictor` applies automatically. The test split never influences
any tuned value.

Backfill an existing checkpoint:
    python -m fiberseg.tools.auto_tune --config <cfg> --checkpoint <best.ckpt> \
        [--run-id <mlflow run id>] [--out-dir <dir, default: checkpoint's folder>]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from ..dataset import _normalize_image, _read_gray, find_pairs
from ..predict_tiles import binarize_prob, load_predictor, predict_prob
from ..tuned_thresholds import TUNED_KEY
from .evaluate_predictions import compute_metrics
from .tune_hysteresis_threshold import hysteresis_counts
from .tune_threshold import _best_index, _counts_at, _threshold_grid, metrics_from_counts

FIXED_STEPS = 99
HYST_LOWS = np.round(np.arange(0.05, 0.951, 0.05), 3)
HYST_HIGHS = np.round(np.arange(0.05, 0.9951, 0.005), 4)
REFERENCE_MODE = "fixed-untuned"
REPORT_METRICS = ["dice", "precision", "recall", "iou", "tol_f1", "cldice"]


def _ridge_response(prob: np.ndarray, cfg) -> np.ndarray:
    from .fiber_gap_repair import enhance_ridges

    return enhance_ridges(
        prob,
        method=cfg.inference.ridge_method,
        sigmas=tuple(cfg.inference.ridge_sigmas),
        device=cfg.inference.ridge_device,
    )


def _split_images(cfg, split: str, model, device):
    """Yield (name, prob, gt) for every image of `split`, one probability map each."""
    pairs = [p for p in find_pairs(cfg.data) if p.split == split]
    if not pairs:
        raise RuntimeError(f"No images in split {split!r}.")
    for i, pair in enumerate(pairs, start=1):
        print(f"  [{split} {i}/{len(pairs)}] {pair.image_path.name}", flush=True)
        prob = predict_prob(_normalize_image(_read_gray(pair.image_path)), model, cfg, device)
        yield pair.image_path.name, prob, _read_gray(pair.mask_path) > 0


def tune_modes(cfg, model, device, modes, *, metric: str = "dice", split: str = "val") -> dict:
    """Tune every mode in `modes` on `split`; returns {"modes": params, "val": stats}."""
    grid = _threshold_grid(FIXED_STEPS)
    acc = {}
    for mode in modes:
        shape = (len(HYST_LOWS), len(HYST_HIGHS)) if mode == "hysteresis" else grid.shape
        acc[mode] = [np.zeros(shape) for _ in range(3)]

    for _, prob, fiber in _split_images(cfg, split, model, device):
        for mode in modes:
            if mode == "fixed":
                counts = _counts_at(prob, fiber, grid)
            elif mode == "ridge":
                counts = _counts_at(_ridge_response(prob, cfg), fiber, grid)
            else:
                counts = hysteresis_counts(prob, fiber, HYST_LOWS, HYST_HIGHS)
            for a, c in zip(acc[mode], counts):
                a += c

    a, b = cfg.train.loss.tversky_alpha, cfg.train.loss.tversky_beta
    params, val_stats = {}, {}
    for mode in modes:
        metrics = metrics_from_counts(*acc[mode], a, b)
        if mode == "hysteresis":
            valid = HYST_LOWS[:, None] < HYST_HIGHS[None, :]
            score = np.where(valid, metrics[metric], -np.inf)
            li, hi = np.unravel_index(int(np.argmax(score)), score.shape)
            params[mode] = {"low": float(HYST_LOWS[li]), "high": float(HYST_HIGHS[hi])}
            val_stats[mode] = {k: float(v[li, hi]) for k, v in metrics.items()}
        else:
            best = _best_index(grid, metrics[metric], FIXED_STEPS)
            val_stats[mode] = {k: float(v[best]) for k, v in metrics.items()}
            if mode == "fixed":
                params[mode] = {"threshold": float(grid[best])}
            else:
                params[mode] = {
                    "ridge_threshold": float(grid[best]),
                    "ridge_method": cfg.inference.ridge_method,
                    "ridge_sigmas": [float(s) for s in cfg.inference.ridge_sigmas],
                }
    return {"modes": params, "val": val_stats}


def _mode_cfg(cfg, mode: str, params: dict):
    """Copy of `cfg` that binarizes with `mode` at the tuned `params`."""
    c = copy.deepcopy(cfg)
    if mode == REFERENCE_MODE:
        c.inference.threshold_mode = "fixed"
        return c
    c.inference.threshold_mode = mode
    p = params[mode]
    if mode == "fixed":
        c.train.threshold = p["threshold"]
    elif mode == "hysteresis":
        c.inference.hysteresis_low, c.inference.hysteresis_high = p["low"], p["high"]
    else:
        c.inference.ridge_threshold = p["ridge_threshold"]
    return c


def evaluate_modes(cfg, model, device, params: dict, *, split: str = "test"):
    """Per-image metrics on `split` for every tuned mode plus the untuned reference.

    Returns ({mode: DataFrame of per-image rows}, {mode: pooled stats}).
    """
    mode_cfgs = {m: _mode_cfg(cfg, m, params) for m in [REFERENCE_MODE, *params]}
    a, b = cfg.train.loss.tversky_alpha, cfg.train.loss.tversky_beta
    rows = {m: [] for m in mode_cfgs}
    for name, prob, fiber in _split_images(cfg, split, model, device):
        for mode, mcfg in mode_cfgs.items():
            m = compute_metrics(binarize_prob(prob, mcfg), fiber, alpha=a, beta=b)
            rows[mode].append({"image": name, "split": split, **m})

    frames, pooled = {}, {}
    for mode, r in rows.items():
        df = pd.DataFrame(r)
        frames[mode] = df
        stats = metrics_from_counts(df.tp.sum(), df.fp.sum(), df.fn.sum(), a, b)
        stats = {k: float(v) for k, v in stats.items()}
        for k in ("tol_f1", "cldice"):
            if k in df:
                stats[f"mean_{k}"] = float(df[k].mean())
        pooled[mode] = stats
    return frames, pooled


def embed_in_checkpoint(checkpoint: str | Path, tuned: dict) -> None:
    """Store `tuned` in the checkpoint file under TUNED_KEY (atomic replace)."""
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt[TUNED_KEY] = tuned
    tmp = checkpoint.with_name(checkpoint.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, checkpoint)


def _log_to_mlflow(client, run_id: str, tuned: dict, files: list[Path]) -> None:
    for mode, p in tuned["modes"].items():
        for k, v in p.items():
            if k in ("ridge_method", "ridge_sigmas"):
                continue
            client.log_param(run_id, f"tuned.{mode}.{k}", round(float(v), 4))
    for phase in ("val", "test"):
        for mode, stats in tuned.get(phase, {}).items():
            for k, v in stats.items():
                name = k.replace("mean_", "")
                if name in REPORT_METRICS or name == "f2":
                    client.log_metric(run_id, f"tuned_{phase}/{mode}/{name}", float(v))
    client.set_tag(run_id, "val_best_threshold_mode", tuned["val_best_mode"])
    for f in files:
        client.log_artifact(run_id, str(f), artifact_path="threshold_tuning")


def auto_tune_checkpoint(cfg, checkpoint, out_dir, *, client=None, run_id=None) -> dict:
    """Tune `cfg.train.auto_tune_modes` on val, score them on test, write outputs, embed
    the tuned values in `checkpoint` and (optionally) log everything to an MLflow run.
    `cfg` is not modified. Returns the stored tuned dict."""
    cfg = copy.deepcopy(cfg)
    modes = list(dict.fromkeys(cfg.train.auto_tune_modes))
    metric = cfg.train.auto_tune_metric
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Auto-tuning thresholds on val (maximize {metric}) for modes: {', '.join(modes)}")
    # Load with a throwaway cfg copy so any values already stored in the checkpoint
    # don't replace the config's own threshold used for the fixed-untuned reference.
    model, device = load_predictor(str(checkpoint), copy.deepcopy(cfg))
    try:
        tuned = tune_modes(cfg, model, device, modes, metric=metric)
        print("Scoring tuned thresholds on test...")
        frames, test_stats = evaluate_modes(cfg, model, device, tuned["modes"])
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tuned = {
        "version": 1,
        "split": "val",
        "metric": metric,
        **tuned,
        "val_best_mode": max(modes, key=lambda m: tuned["val"][m][metric]),
        "test": test_stats,
        "config_threshold": float(cfg.train.threshold),
    }

    files = []
    json_path = out_dir / "tuned_thresholds.json"
    json_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    files.append(json_path)
    for mode, df in frames.items():
        path = out_dir / f"test_metrics_{mode}.csv"
        df.to_csv(path, index=False)
        files.append(path)

    embed_in_checkpoint(checkpoint, tuned)
    if client is not None and run_id is not None:
        _log_to_mlflow(client, run_id, tuned, files)

    _print_summary(tuned, metric)
    return tuned


def _print_summary(tuned: dict, metric: str) -> None:
    print("-" * 80)
    print(f"{'mode':14s} {'params':34s} {'val ' + metric:>9s} {'test dice':>9s} "
          f"{'test prec':>9s} {'test rec':>9s}")
    rows = [(REFERENCE_MODE, f"threshold={tuned['config_threshold']:.3f} (untuned)")]
    rows += [(m, ", ".join(f"{k}={v:.3f}" for k, v in p.items() if isinstance(v, float)))
             for m, p in tuned["modes"].items()]
    for mode, desc in rows:
        val = tuned["val"].get(mode, {}).get(metric)
        t = tuned["test"][mode]
        val_s = f"{val:9.4f}" if val is not None else f"{'-':>9s}"
        print(f"{mode:14s} {desc:34s} {val_s} {t['dice']:9.4f} {t['precision']:9.4f} "
              f"{t['recall']:9.4f}")
    print(f"Best mode on val: {tuned['val_best_mode']}  (stored in checkpoint; "
          "load_predictor applies all tuned values)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Tune thresholds on val for an existing checkpoint, score on test, "
        "and store the tuned values in the checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None,
                        help="Where to write tuned_thresholds.json and per-mode test CSVs "
                        "(default: the checkpoint's folder).")
    parser.add_argument("--run-id", default=None,
                        help="MLflow run id to log the results into (optional).")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=["fixed", "hysteresis", "ridge"],
                        help="Override train.auto_tune_modes.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.modes:
        cfg.train.auto_tune_modes = args.modes
    client = None
    if args.run_id:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = MlflowClient()
    ckpt = Path(args.checkpoint)
    auto_tune_checkpoint(cfg, ckpt, args.out_dir or ckpt.parent, client=client, run_id=args.run_id)


if __name__ == "__main__":
    main()
