# sweep.py
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from typing import Any

import lightning.pytorch as pl
import mlflow

from .config import load_config
from .train import _run_single_training


def set_nested(obj: Any, dotted_key: str, value: Any):
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run a simple grid search from the config's sweep section."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    base = load_config(args.config)
    if not base.sweep:
        raise ValueError(
            "No sweep section found in config. Add e.g. sweep: {'data.patch_size': [256,512]}."
        )

    pl.seed_everything(base.data.seed, workers=True)

    keys = list(base.sweep.keys())
    values = [base.sweep[k] for k in keys]

    total_runs = 1
    for v in values:
        total_runs *= len(v)

    print("=" * 80)
    print(f"Starting sweep with {total_runs} combinations")
    print("Parameter grid:")
    for run_idx, combo in enumerate(itertools.product(*values), start=1):
        combo_summary = {key: _format_value(value) for key, value in zip(keys, combo)}
        print(f"  {run_idx:03d}/{total_runs}: {json.dumps(combo_summary, sort_keys=True)}")
    print("=" * 80, flush=True)

    for run_idx, combo in enumerate(itertools.product(*values), start=1):
        cfg = copy.deepcopy(base)
        sweep_params: dict[str, Any] = {}
        suffix_parts: list[str] = []

        for key, value in zip(keys, combo):
            set_nested(cfg, key, value)
            sweep_params[key] = value
            suffix_parts.append(f"{key.split('.')[-1]}={_format_value(value)}")

        version_label = f"sweep-{run_idx:03d}"
        param_summary = " | ".join(suffix_parts) if suffix_parts else "default"
        cfg.mlflow.run_name = f"{version_label} | {param_summary}"
        cfg.mlflow.experiment_name = cfg.mlflow.experiment_name or "fiber-sem-segmentation"

        print("=" * 80)
        print(f"Starting sweep run {run_idx}/{total_runs}")
        print(f"Version: {version_label}")
        print(f"Run name: {cfg.mlflow.run_name}")
        print(f"Sweep params: {json.dumps(sweep_params, sort_keys=True)}")
        print(f"Learning rate: {cfg.train.learning_rate}")
        print(f"Batch size: {cfg.data.batch_size}")
        print(f"Patch size: {cfg.data.patch_size}")
        print("=" * 80, flush=True)
        sys.stdout.flush()

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        experiment_name = cfg.mlflow.experiment_name or "fiber-sem-segmentation"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=cfg.mlflow.run_name, experiment_id=None):
            mlflow.log_param("sweep_version", version_label)
            mlflow.log_param("sweep_total_runs", total_runs)
            mlflow.log_param("sweep_params", json.dumps(sweep_params, sort_keys=True))
            mlflow.log_param("sweep_run_index", run_idx)
            mlflow.log_param("sweep_param_summary", param_summary)
            _run_single_training(cfg)


if __name__ == "__main__":
    main()
