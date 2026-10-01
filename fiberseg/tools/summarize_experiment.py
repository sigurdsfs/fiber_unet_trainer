# summarize_experiment.py
"""One table for a multi-arm, multi-seed MLflow experiment (e.g. configs/proxy/scratch_ablation).

Reads every FINISHED run of `--experiment`, groups runs by their `arm` tag (falling
back to the run name before " | seed=..."), and for every threshold mode that
auto-tuning scored (fixed-untuned / fixed / hysteresis / ridge) reports:

1. arm x mode: mean +- sd over seeds of the pooled test dice/precision/recall
   (MLflow metrics tuned_test/<mode>/<metric>, tuned on val - test never picks anything);
2. each arm vs the baseline arm, per mode: a paired bootstrap over test images
   (tools.compare_runs, per-image test_metrics_<mode>.csv files from each run's
   checkpoint_dir, averaged across seeds first). Trust a difference only when its
   CI excludes 0.

Run:
    python -m fiberseg.tools.summarize_experiment --experiment proxy-r34-scratch-ablation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from .compare_runs import compare, load_side

MODES = ["fixed-untuned", "fixed", "hysteresis", "ridge"]
METRICS = ["dice", "precision", "recall"]
REPO = Path(__file__).resolve().parents[2]


def _arm(run) -> str:
    return run.data.tags.get("arm") or run.info.run_name.split(" | ")[0]


def _csv(run, mode: str) -> Path | None:
    ckpt_dir = run.data.params.get("checkpoint_dir")
    if not ckpt_dir:
        return None
    for root in (Path.cwd(), REPO):
        p = root / ckpt_dir / f"test_metrics_{mode}.csv"
        if p.is_file():
            return p
    return None


def collect(experiment: str, tracking_uri: str) -> list:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"No MLflow experiment named {experiment!r}.")
    runs = client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'",
                              max_results=1000)
    if not runs:
        raise SystemExit(f"No finished runs in {experiment!r} yet.")
    return runs


def arm_mode_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        for mode in MODES:
            vals = {m: r.data.metrics.get(f"tuned_test/{mode}/{m}") for m in METRICS}
            if vals["dice"] is None:
                continue
            rows.append({"arm": _arm(r), "mode": mode, "seed": r.data.tags.get("seed"), **vals})
    if not rows:
        raise SystemExit("No runs have tuned_test/* metrics (auto-tuning off or failed?).")
    df = pd.DataFrame(rows)
    agg = df.groupby(["arm", "mode"]).agg(
        seeds=("dice", "size"),
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_sd": (m, "std") for m in METRICS},
    ).reset_index()
    return agg.sort_values(["mode", "dice_mean"], ascending=[True, False])


def vs_baseline(runs, baseline_arm: str, n_boot: int) -> pd.DataFrame:
    by_arm: dict[str, list] = {}
    for r in runs:
        by_arm.setdefault(_arm(r), []).append(r)
    if baseline_arm not in by_arm:
        raise SystemExit(f"Baseline arm {baseline_arm!r} not found; arms: {sorted(by_arm)}")

    rows = []
    for mode in MODES:
        a_files = [p for r in by_arm[baseline_arm] if (p := _csv(r, mode))]
        if not a_files:
            continue
        a, _ = load_side([str(p) for p in a_files], "test", None)
        for arm, arm_runs in sorted(by_arm.items()):
            if arm == baseline_arm:
                continue
            b_files = [p for r in arm_runs if (p := _csv(r, mode))]
            if not b_files:
                continue
            b, _ = load_side([str(p) for p in b_files], "test", None)
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            table = compare(a.loc[common], b.loc[common], ["dice", "precision", "recall"], n_boot)
            for rec in table.to_dict("records"):
                rows.append({"mode": mode, "arm": arm, "seeds_A": len(a_files),
                             "seeds_B": len(b_files), **rec})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize a multi-arm MLflow experiment.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--baseline-arm", default="00 baseline")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--out-dir", default=None,
                        help="Write arm_mode_summary.csv and vs_baseline.csv here.")
    args = parser.parse_args()

    runs = collect(args.experiment, args.tracking_uri)
    summary = arm_mode_table(runs)
    comparison = vs_baseline(runs, args.baseline_arm, args.n_boot)

    fmt = {"display.float_format": "{:.4f}".format, "display.width": 200,
           "display.max_rows": 500}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print(f"\n{args.experiment}: {len(runs)} finished runs")
        print("\n== Test metrics per arm and threshold mode (mean/sd over seeds; "
              "thresholds tuned on val) ==")
        print(summary.to_string(index=False))
        if not comparison.empty:
            print(f"\n== Paired bootstrap vs {args.baseline_arm!r} (test images, "
                  "seeds averaged per image) ==")
            cols = ["mode", "arm", "metric", "A", "B", "B-A", "ci_low", "ci_high",
                    "boot_P(B-A<=0)", "seeds_B"]
            print(comparison[cols].to_string(index=False))
    print("\nRead: a difference is real only when its CI excludes 0. '02 data-50nm' is "
          "scored in 50 nm/px space, so its per-image numbers are comparable on the same "
          "images but not pixel-for-pixel.")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "arm_mode_summary.csv", index=False)
        comparison.to_csv(out / "vs_baseline.csv", index=False)
        print(f"Wrote {out / 'arm_mode_summary.csv'} and {out / 'vs_baseline.csv'}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
