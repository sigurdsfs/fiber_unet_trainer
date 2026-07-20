# config.py
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _validate_split_fractions(split: dict[str, float]) -> None:
    total = sum(float(split.get(k, 0.0)) for k in ("train", "val", "test"))
    if not 0.999 <= total <= 1.001:
        raise ValueError("Config split fractions must sum to 1")


@dataclass
class DataConfig:
    images_dir: str
    masks_dir: str
    image_glob: str = "*.tif"
    mask_pattern: str = "{stem}_mask.tif"
    patch_size: int | list[int] = 512
    stride: int | list[int] | None = None
    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    seed: int = 42
    min_foreground_fraction: float = 0.0
    keep_empty_probability: float = 1.0
    num_workers: int = 4
    batch_size: int = 4

    # Directory for precomputed, percentile-normalized full-image caches (.npy).
    # Avoids re-normalizing whole source images on every tile access. Shared across
    # runs/sweeps as long as source images don't change. Defaults to
    # "./cache/normalized_images" (relative to the working directory) when null.
    cache_dir: str | None = None

    # Number of channels returned by the dataset.
    # Use 1 for normal grayscale models.
    # Use 3 for MicroNet, where grayscale SEM images are copied into RGB-like channels.
    image_channels: int = 1


@dataclass
class ModelConfig:
    architecture: str = "Unet"
    encoder_name: str = "resnet34"
    encoder_weights: str | None = None
    in_channels: int = 1
    classes: int = 1


@dataclass
class LossConfig:
    name: str = "bce_dice"

    # Focal Tversky settings. beta > alpha penalizes missed fibers more strongly.
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_gamma: float = 0.75


@dataclass
class SchedulerConfig:
    # Use "reduce_on_plateau" for adaptive LR reduction, "cosine" for cosine
    # annealing, or "none"/"off"/"false" to disable scheduling entirely.
    name: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    # Only attached when train.max_epochs >= 10.
    patience: int = 15
    min_delta: float = 0.001


@dataclass
class TrainConfig:
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32-true"
    threshold: float = 0.5

    # When set, the encoder gets its own param group at learning_rate *
    # encoder_lr_ratio while the rest of the model (decoder/head) keeps
    # learning_rate. Intended for pretrained encoders (encoder_weights:
    # "imagenet"/"micronet") so the pretrained features are fine-tuned gently
    # while the randomly-initialized decoder learns faster. Leave null for
    # from-scratch training (encoder_weights: null), where a single LR is
    # correct and this has no effect.
    encoder_lr_ratio: float | None = None

    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    matmul_precision: str | None = "medium"
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    deterministic: bool = False

    # Path to a .ckpt file to resume training from (restores model, optimizer,
    # scheduler, and epoch count). Leave null to start a fresh run.
    resume_from_checkpoint: str | None = None

    # Bottleneck diagnostics. profiler: null (off), "simple", "advanced", or
    # "pytorch" (passed straight through to lightning.pytorch.Trainer(profiler=...)).
    # limit_train_batches/limit_val_batches cap batches per epoch (int count or
    # float fraction) so a diagnostic run doesn't need a full epoch to finish;
    # leave null to use the whole dataset.
    profiler: str | None = None
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None

    # Folder to export the best checkpoint to (as TorchScript) right after training
    # finishes. The exported subfolder/model name is derived from the MLflow
    # experiment and run name. Leave null to skip export.
    model_export_path: str | None = None


@dataclass
class MlflowConfig:
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "fiber-sem-segmentation"
    run_name: str | None = None


@dataclass
class LoggingConfig:
    log_prediction_images: bool = True
    prediction_max_images: int = 4
    prediction_artifact_dir: str = "predictions/best_model_test"


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    augmentations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sweep: dict[str, list[Any]] = field(default_factory=dict)

    # Read independently by fiberseg.tools.foreground_filter_sweep (not validated
    # into a dataclass - see configs/foreground_filter_sweep_example.yaml). Kept
    # here only so load_config's top-level key check doesn't reject it.
    foreground_filter_sweep: dict[str, Any] = field(default_factory=dict)


_KNOWN_TOP_LEVEL_KEYS = {f.name for f in dataclasses.fields(AppConfig)}


def _dataclass_from_dict(cls, values: dict[str, Any] | None, *, context: str = ""):
    """Build `cls` from a dict, raising on any key that isn't a real field.

    Recurses into nested dataclass fields (detected via `field(default_factory=...)`
    pointing at another dataclass) so e.g. an unknown key under `train.loss` is
    caught just as loudly as one directly under `train`. Deliberately strict: a
    typo'd or stale config key must fail loudly, not silently fall back to a
    default the user never intended.
    """
    values = values or {}
    fields_by_name = {f.name: f for f in dataclasses.fields(cls)}

    unknown = sorted(set(values) - set(fields_by_name))
    if unknown:
        where = context or cls.__name__
        raise ValueError(f"Unknown config key(s) in {where!r}: {unknown}")

    kwargs: dict[str, Any] = {}
    for name, f in fields_by_name.items():
        if name not in values:
            continue
        value = values[name]
        nested_cls = f.default_factory if dataclasses.is_dataclass(f.default_factory) else None
        if nested_cls is not None and isinstance(value, dict):
            nested_context = f"{context}.{name}" if context else name
            value = _dataclass_from_dict(nested_cls, value, context=nested_context)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = raw or {}

    unknown_sections = sorted(set(raw) - _KNOWN_TOP_LEVEL_KEYS)
    if unknown_sections:
        raise ValueError(f"Unknown top-level config section(s): {unknown_sections}")

    if "data" not in raw:
        raise ValueError("Config must contain a 'data:' section.")

    data_cfg = raw.get("data", {}) or {}
    if "images_dir" not in data_cfg:
        raise ValueError("Config data section must include 'images_dir'.")
    if "masks_dir" not in data_cfg:
        raise ValueError("Config data section must include 'masks_dir'.")

    split = data_cfg.get("split", {}) or {}
    defaults = {"train": 0.7, "val": 0.15, "test": 0.15}
    data_cfg = dict(data_cfg)
    data_cfg["split"] = {k: float(split.get(k, defaults[k])) for k in defaults}
    _validate_split_fractions(data_cfg["split"])

    return AppConfig(
        data=_dataclass_from_dict(DataConfig, data_cfg, context="data"),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {}), context="model"),
        train=_dataclass_from_dict(TrainConfig, raw.get("train", {}), context="train"),
        mlflow=_dataclass_from_dict(MlflowConfig, raw.get("mlflow", {}), context="mlflow"),
        logging=_dataclass_from_dict(LoggingConfig, raw.get("logging", {}), context="logging"),
        augmentations=raw.get("augmentations", {}) or {},
        sweep=raw.get("sweep", {}) or {},
        foreground_filter_sweep=raw.get("foreground_filter_sweep", {}) or {},
    )


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}

    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]

    return obj