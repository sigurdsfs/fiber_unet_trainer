# train.py
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import mlflow
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from .callbacks import BestModelPredictionImageLogger, PeriodicPredictionImageLogger
from .config import AppConfig, TrainConfig, load_config, to_dict
from .dataset import FiberDataModule, split_filenames
from .lit_module import FiberSegmentationLitModule
from .tools.export_torchscript import export_torchscript

def _set_nested(obj: object, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cur = obj
    for part in parts[:-1]:
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise AttributeError(f"Could not set nested value for {dotted_key}")
    setattr(cur, parts[-1], value)


def _format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _iter_sweep_combinations(cfg: AppConfig) -> list[tuple[dict[str, object], list[str]]]:
    if not getattr(cfg, "sweep", None):
        return []

    keys = list(cfg.sweep.keys())
    values = [cfg.sweep[k] for k in keys]
    combos: list[tuple[dict[str, object], list[str]]] = []
    for combo in itertools.product(*values):
        combo_map = {key: value for key, value in zip(keys, combo)}
        suffix_parts = [f"{key.split('.')[-1]}={_format_value(value)}" for key, value in combo_map.items()]
        combos.append((combo_map, suffix_parts))
    return combos


def _expand_sweep_configs(cfg: AppConfig) -> list[AppConfig]:
    if not getattr(cfg, "sweep", None):
        return [cfg]

    expanded: list[AppConfig] = []
    for combo_map, suffix_parts in _iter_sweep_combinations(cfg):
        new_cfg = copy.deepcopy(cfg)
        for key, value in combo_map.items():
            _set_nested(new_cfg, key, value)
        new_cfg.mlflow.run_name = " | ".join(suffix_parts)
        expanded.append(new_cfg)
    return expanded


_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_for_path(name: str) -> str:
    cleaned = _INVALID_PATH_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
    return cleaned or "run"


def _print_sweep_summary(cfg: AppConfig) -> None:
    combos = _iter_sweep_combinations(cfg)
    if not combos:
        return

    total_runs = len(combos)
    keys = list(cfg.sweep.keys())

    print("=" * 80)
    print(f"Sweep configuration detected: {total_runs} combinations")
    print("Sweep grid:")
    for run_idx, (_, suffix_parts) in enumerate(combos, start=1):
        print(f"  {run_idx:03d}/{total_runs}: {', '.join(suffix_parts)}")
    print("=" * 80)


def _silence_torch_flop_counter_warnings() -> None:
    for logger_name in (
        "torch.utils.flop_counter",
        "torch._dynamo",
        "torch._inductor",
        "triton",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        logger.disabled = True


_silence_torch_flop_counter_warnings()


def resolve_trainer_settings(
    cfg: AppConfig | TrainConfig | object,
    *,
    cuda_available: bool | None = None,
):
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    train_cfg = cfg.train if hasattr(cfg, "train") else cfg

    accelerator = getattr(train_cfg, "accelerator", "auto")
    devices = getattr(train_cfg, "devices", "auto")
    precision = getattr(train_cfg, "precision", "32-true")

    if accelerator == "auto":
        accelerator = "gpu" if cuda_available else "cpu"
    elif accelerator == "gpu" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available on this machine.")

    if accelerator == "gpu" and precision == "32-true":
        precision = "16-mixed"

    return accelerator, devices, precision


def run_training(cfg: AppConfig):
    pl.seed_everything(cfg.data.seed, workers=True)

    if getattr(cfg, "sweep", None):
        _print_sweep_summary(cfg)

    configs_to_run = _expand_sweep_configs(cfg)
    if len(configs_to_run) > 1:
        print(f"Expanding sweep into {len(configs_to_run)} training runs")

    for idx, run_cfg in enumerate(configs_to_run, start=1):
        if len(configs_to_run) > 1:
            print(f"\n=== Running sweep combination {idx}/{len(configs_to_run)} ===")
        _run_single_training(run_cfg)


def _log_filtered_tile_stats(cfg: AppConfig, datamodule: FiberDataModule, logger: MLFlowLogger | None) -> None:
    filtered_count = datamodule.filtered_tiles_count
    total_candidates = len(datamodule.train_ds.tiles) + filtered_count
    fraction = datamodule.filtered_tiles_fraction if total_candidates else 0.0

    print(
        "Filtered tiles: "
        f"{filtered_count} / {total_candidates} ({fraction:.2%})"
    )

    try:
        mlflow_cfg = getattr(cfg, "mlflow", None)
        tracking_uri = getattr(mlflow_cfg, "tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        metrics = {
            "filtered_tiles_count": float(filtered_count),
            "filtered_tiles_fraction": float(fraction),
        }
        if logger is not None:
            logger.log_metrics(metrics=metrics, step=0)
        else:
            with mlflow.start_run():
                mlflow.log_metrics(metrics)
    except Exception:
        pass


def _save_split_files(cfg: AppConfig, checkpoint_dir: Path, logger: MLFlowLogger) -> None:
    """Write train/validation/test image filenames to checkpoint_dir and log them to MLflow."""
    result = split_filenames(cfg.data)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "split_files.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        "Data split: "
        f"train={len(result['Train'])}, validation={len(result['validation'])}, "
        f"test={len(result['Test'])} files"
    )

    try:
        logger.experiment.log_artifact(
            run_id=logger.run_id,
            local_path=str(path),
            artifact_path="data_split",
        )
    except Exception:
        pass


def _run_single_training(cfg: AppConfig):

    if cfg.train.matmul_precision:
        torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    # Tile shapes are constant per run (always padded to patch_size), so cuDNN
    # can safely autotune and cache the fastest conv algorithm for that shape.
    torch.backends.cudnn.benchmark = True

    run_name = cfg.mlflow.run_name or (
        f"{cfg.model.architecture}-{cfg.model.encoder_name}-"
        f"p{cfg.data.patch_size}-lr{cfg.train.learning_rate}"
    )

    print("=" * 80)
    print("Starting training run")
    print(f"Run name: {run_name}")

    if getattr(cfg, "sweep", None):
        print("Current selection:")
        for key in cfg.sweep:
            value = None
            parts = key.split(".")
            current = cfg
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                    continue
                if isinstance(current, dict) and part in current:
                    current = current[part]
                    continue
                current = None
                break
            if current is not None:
                print(f"  - {key} = {_format_value(current)}")
    else:
        print(
            "Config summary: "
            f"patch_size={cfg.data.patch_size}, stride={cfg.data.stride}, "
            f"lr={cfg.train.learning_rate}, encoder={cfg.model.encoder_name}, "
            f"loss={cfg.train.loss.name}, batch_size={cfg.data.batch_size}"
        )
    print("=" * 80)

    resume_ckpt = cfg.train.resume_from_checkpoint or None
    if resume_ckpt and not Path(resume_ckpt).is_file():
        raise FileNotFoundError(f"train.resume_from_checkpoint not found: {resume_ckpt}")

    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=run_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = (
        Path("checkpoints")
        / _sanitize_for_path(cfg.mlflow.experiment_name)
        / f"{_sanitize_for_path(run_name)}_{timestamp}"
    )
    print(f"Checkpoint dir: {checkpoint_dir}")
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    logger.log_hyperparams({"checkpoint_dir": str(checkpoint_dir), "resumed_from": resume_ckpt or ""})
    _save_split_files(cfg, checkpoint_dir, logger)

    datamodule = FiberDataModule(cfg.data, cfg.augmentations)
    datamodule.setup("fit")
    _log_filtered_tile_stats(cfg, datamodule, logger)
    model = FiberSegmentationLitModule(cfg.model, cfg.train)

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val/tversky",
        mode="max",
        filename="best-{epoch:03d}-{val_tversky:.4f}",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks = [
        best_checkpoint_callback,
        ModelCheckpoint(dirpath=str(checkpoint_dir), filename="last", save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.logging.log_prediction_images:
        callbacks.append(
            PeriodicPredictionImageLogger(
                max_images=10,
                threshold=cfg.train.threshold,
                artifact_dir=f"{cfg.logging.prediction_artifact_dir}/periodic",
                every_n_epochs=15,
                max_candidate_samples=500,
            )
        )

        callbacks.append(
            BestModelPredictionImageLogger(
                max_images=cfg.logging.prediction_max_images,
                threshold=cfg.train.threshold,
                artifact_dir=f"{cfg.logging.prediction_artifact_dir}/best_model",
                max_candidate_samples=500,
            )
        )

    if cfg.train.max_epochs >= 10:
        callbacks.append(
            EarlyStopping(
                monitor="val/tversky",
                mode="max",
                patience=cfg.train.early_stopping.patience,
                min_delta=cfg.train.early_stopping.min_delta,
            )
        )

    accelerator, devices, precision = resolve_trainer_settings(cfg)

    trainer_kwargs = {}
    if cfg.train.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cfg.train.limit_train_batches
    if cfg.train.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = cfg.train.limit_val_batches

    if cfg.train.profiler:
        print(f"Profiler enabled: {cfg.train.profiler!r} (report printed at the end of this run)")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        deterministic=cfg.train.deterministic,
        enable_model_summary=True,
        profiler=cfg.train.profiler,
        **trainer_kwargs,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    _log_filtered_tile_stats(cfg, datamodule, logger)

    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    try:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.log_param("python_env", os.environ.get("CONDA_DEFAULT_ENV", "unknown"))
        mlflow.log_param("python_executable", os.environ.get("CONDA_PREFIX", "unknown"))
        mlflow.log_param("git_sha", os.environ.get("GIT_COMMIT", "unknown"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_config.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
            mlflow.log_artifact(str(path), artifact_path="config")
    except Exception:
        pass

    if cfg.train.model_export_path:
        best_ckpt = best_checkpoint_callback.best_model_path
        if not best_ckpt:
            print("model_export_path is set but no best checkpoint was saved; skipping export.")
        else:
            export_name = _sanitize_for_path(f"{cfg.mlflow.experiment_name}_{run_name}")
            export_dir = Path(cfg.train.model_export_path) / export_name
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    export_config_path = Path(tmp) / "resolved_config.yaml"
                    with open(export_config_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
                    export_torchscript(
                        config_path=export_config_path,
                        checkpoint_path=best_ckpt,
                        out_dir=export_dir,
                        model_name=export_name,
                    )
                print(f"Exported model to: {export_dir.resolve()}")
            except Exception as exc:
                print(f"Model export failed: {exc}")

    return trainer.callback_metrics


def main():
    parser = argparse.ArgumentParser(description="Train binary SEM fiber segmentation model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a .ckpt file to resume training from. Overrides train.resume_from_checkpoint.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Attach a Lightning profiler to diagnose bottlenecks. Overrides train.profiler.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg.train.resume_from_checkpoint = args.resume
    if args.profile:
        cfg.train.profiler = args.profile
    run_training(cfg)


if __name__ == "__main__":
    main()