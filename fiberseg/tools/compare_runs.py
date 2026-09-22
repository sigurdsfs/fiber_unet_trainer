# compare_runs.py
"""Is run B really better than run A? Paired bootstrap over per-image metrics.

With ~45 test images, a single run's mean dice has a 95% interval roughly +-0.045 wide,
so a 0.02-0.03 difference between two single-seed runs is usually noise. This compares
two sets of `metrics.csv` files (from `predict_all.py`, `evaluate_predictions.py` or
`postprocess_masks.py`) image by image - pairing removes the between-image variance that
dominates an unpaired comparison - and bootstraps the difference.

Each side can take several CSVs (e.g. the same config trained with train.seed 1/2/3):
per-image metrics are averaged across that side's files first, and the seed-to-seed
spread of the mean is reported so you can see how much of a difference training noise
alone produces.

Run:
    python -m fiberseg.tools.compare_runs \
        --a predictions/baseline/metrics.csv \
        --b predictions/balanced_s1/metrics.csv predictions/balanced_s2/metrics.csv \
        --split test

Only compare runs evaluated on the same split definition (same images_dir/seed/split
fractions) - images present on only one side are dropped with a warning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METRICS = ["dice", "precision", "recall", "tol_f1", "cldice", "iou"]
_SPLIT_ALIASES = {"val": {"val", "validation"}, "test": {"test"}, "train": {"train"}}


def _metric_prefix(columns, prefix: str | None) -> str:
    """predict_all writes raw_*/post_*; evaluate_predictions writes bare names."""
    if prefix is not None:
        return prefix
    return "raw_" if "raw_dice" in columns else ""


def load_side(
    paths: list[str], split: str | None, prefix: str | None
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Per-image metrics averaged across `paths` (bare metric names, indexed by image),
    plus each file's own frame for the seed-spread report."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        pre = _metric_prefix(df.columns, prefix)
        if split and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower().isin(_SPLIT_ALIASES.get(split, {split}))]
        cols = {c: c[len(pre):] for c in df.columns if pre and c.startswith(pre)}
        df = df.rename(columns=cols).set_index("image")
        frames.append(df)
    common = sorted(set.intersection(*(set(f.index) for f in frames)))
    if len(common) < max(len(f) for f in frames):
        print(f"Warning: {Path(paths[0]).parent.name}: files disagree on images; "
              f"keeping the {len(common)} present in all.")
    numeric = [f.loc[common].select_dtypes("number") for f in frames]
    return sum(numeric) / len(numeric), [f.loc[common] for f in frames]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """Mean of (b - a), its 95% bootstrap CI, and the bootstrap share of resamples where
    the mean difference is <= 0 (small => B reliably better; large => B reliably worse)."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((boot <= 0).mean())


def pooled_dice_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 10_000, seed: int = 0):
    """Same as `paired_bootstrap` but for micro (pixel-pooled) dice from tp/fp/fn columns,
    resampling images jointly for both runs."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))

    def dice(df, ix):
        tp, fp, fn = (df[c].to_numpy()[ix].sum(axis=-1) for c in ("tp", "fp", "fn"))
        return 2 * tp / (2 * tp + fp + fn + 1e-8)

    full = np.arange(len(a))
    point = float(dice(b, full) - dice(a, full))
    boot = dice(b, idx) - dice(a, idx)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), float((boot <= 0).mean())


def compare(
    a: pd.DataFrame, b: pd.DataFrame, metrics: list[str], n_boot: int = 10_000
) -> pd.DataFrame:
    rows = []
    for m in metrics:
        if m not in a.columns or m not in b.columns:
            continue
        va, vb = a[m].to_numpy(float), b[m].to_numpy(float)
        diff, lo, hi, p_le0 = paired_bootstrap(va, vb, n_boot)
        rows.append({
            "metric": f"mean {m}", "A": va.mean(), "B": vb.mean(), "B-A": diff,
            "ci_low": lo, "ci_high": hi, "boot_P(B-A<=0)": p_le0,
            "images_B_better": int((vb > va).sum()), "n": len(va),
        })
    if all(c in a.columns and c in b.columns for c in ("tp", "fp", "fn")):
        diff, lo, hi, p_le0 = pooled_dice_bootstrap(a, b, n_boot)
        tpa, fpa, fna = (a[c].sum() for c in ("tp", "fp", "fn"))
        tpb, fpb, fnb = (b[c].sum() for c in ("tp", "fp", "fn"))
        rows.append({
            "metric": "pooled dice", "A": 2 * tpa / (2 * tpa + fpa + fna),
            "B": 2 * tpb / (2 * tpb + fpb + fnb), "B-A": diff, "ci_low": lo, "ci_high": hi,
            "boot_P(B-A<=0)": p_le0, "images_B_better": np.nan, "n": len(a),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of two runs.")
    parser.add_argument("--a", nargs="+", required=True,
                        help="metrics.csv file(s) for run A (reference).")
    parser.add_argument("--b", nargs="+", required=True, help="metrics.csv file(s) for run B.")
    parser.add_argument("--split", default="test", help="train/val/test, or 'all' (default: test).")
    parser.add_argument("--prefix", default=None,
                        help="Column prefix to compare, e.g. raw_ or post_ "
                        "(default: raw_ if present).")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out", default=None, help="Optional CSV for the comparison table.")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    a, a_runs = load_side(args.a, split, args.prefix)
    b, b_runs = load_side(args.b, split, args.prefix)
    common = sorted(set(a.index) & set(b.index))
    if not common:
        raise SystemExit("No images in common between A and B.")
    if len(common) < max(len(a), len(b)):
        print(f"Warning: A has {len(a)} images, B has {len(b)}; "
              f"comparing the {len(common)} shared.")
    a, b = a.loc[common], b.loc[common]

    table = compare(a, b, args.metrics, args.n_boot)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(f"\nSplit={args.split!r}, {len(common)} paired images, "
              f"A={len(args.a)} file(s), B={len(args.b)} file(s)")
        print(table.to_string(index=False))

    for name, runs in (("A", a_runs), ("B", b_runs)):
        if len(runs) > 1 and "dice" in runs[0].columns:
            means = [r.loc[common, "dice"].mean() for r in runs]
            print(f"{name}: mean dice per file {np.round(means, 4).tolist()} "
                  f"-> seed-to-seed spread {max(means) - min(means):.4f}")

    print("\nRead: a difference is only trustworthy when its CI excludes 0 (boot_P near 0 "
          "or 1). Otherwise treat A and B as tied and prefer the cheaper one.")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
