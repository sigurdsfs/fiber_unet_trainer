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


_VALID_NORMALIZATIONS = {"minmax", "imagenet", "dataset"}


def _validate_normalization(data_cfg: dict[str, Any]) -> None:
    """Validate image_normalization and, for "dataset", any provided norm_mean/std.

    Deliberately does NOT require norm_mean/std to be present for "dataset" mode, so
    the compute_dataset_stats tool can load a stats-less config and fill them in.
    The missing-stats error is raised later (at training/inference), pointing at the
    tool. Here we only reject an unknown mode and length-mismatched stats.
    """
    mode = data_cfg.get("image_normalization", "minmax")
    if mode not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown image_normalization {mode!r}. Use one of {sorted(_VALID_NORMALIZATIONS)}."
        )

    channels = int(data_cfg.get("image_channels", 1))
    for key in ("norm_mean", "norm_std"):
        value = data_cfg.get(key)
        if value is not None and len(value) != channels:
            raise ValueError(
                f"data.{key} has length {len(value)} but data.image_channels is {channels}; "
                "they must match."
            )


_VALID_TILE_SAMPLING = {"static", "weighted"}


def _validate_sampling(data_cfg: dict[str, Any]) -> None:
    mode = data_cfg.get("tile_sampling", "static")
    if mode not in _VALID_TILE_SAMPLING:
        raise ValueError(
            f"Unknown tile_sampling {mode!r}. Use one of {sorted(_VALID_TILE_SAMPLING)}."
        )
    if float(data_cfg.get("negative_ratio", 1.0)) < 0:
        raise ValueError("data.negative_ratio must be >= 0.")
    hnf = float(data_cfg.get("hard_negative_fraction", 0.0))
    if not 0.0 <= hnf <= 1.0:
        raise ValueError("data.hard_negative_fraction must be in [0, 1].")
    if int(data_cfg.get("hard_negative_warmup_epochs", 3)) < 0:
        raise ValueError("data.hard_negative_warmup_epochs must be >= 0.")
    ratio_final = data_cfg.get("negative_ratio_final")
    if ratio_final is not None and float(ratio_final) < 0:
        raise ValueError("data.negative_ratio_final must be >= 0.")
    if int(data_cfg.get("negative_ratio_anneal_epochs", 0)) < 0:
        raise ValueError("data.negative_ratio_anneal_epochs must be >= 0.")


_VALID_THRESHOLD_MODES = {"fixed", "hysteresis", "ridge"}
_VALID_RIDGE_METHODS = {"frangi", "sato"}
_VALID_RIDGE_DEVICES = {"cuda", "cpu"}


def _validate_inference(inf_cfg: dict[str, Any]) -> None:
    mode = inf_cfg.get("threshold_mode", "fixed")
    if mode not in _VALID_THRESHOLD_MODES:
        raise ValueError(
            f"Unknown threshold_mode {mode!r}. Use one of {sorted(_VALID_THRESHOLD_MODES)}."
        )
    method = inf_cfg.get("ridge_method", "frangi")
    if method not in _VALID_RIDGE_METHODS:
        raise ValueError(
            f"Unknown ridge_method {method!r}. Use one of {sorted(_VALID_RIDGE_METHODS)}."
        )
    device = inf_cfg.get("ridge_device")
    if device is not None and device not in _VALID_RIDGE_DEVICES:
        raise ValueError(
            f"Unknown ridge_device {device!r}. Use one of "
            f"{sorted(_VALID_RIDGE_DEVICES)}, or null to pick automatically."
        )
    if int(inf_cfg.get("batch_size", 8)) < 1:
        raise ValueError("inference.batch_size must be >= 1.")
    sigmas = inf_cfg.get("ridge_sigmas")
    if sigmas is not None:
        if not sigmas:
            raise ValueError("inference.ridge_sigmas must not be empty.")
        if any(float(s) <= 0 for s in sigmas):
            raise ValueError("inference.ridge_sigmas must all be > 0.")


def _validate_loss(loss_cfg: dict[str, Any]) -> None:
    if float(loss_cfg.get("focal_bce_weight", 0.0)) < 0:
        raise ValueError("train.loss.focal_bce_weight must be >= 0.")
    alpha = float(loss_cfg.get("focal_bce_alpha", 0.25))
    if not 0.0 < alpha < 1.0:
        raise ValueError("train.loss.focal_bce_alpha must be in (0, 1).")
    if float(loss_cfg.get("focal_bce_gamma", 2.0)) < 0:
        raise ValueError("train.loss.focal_bce_gamma must be >= 0.")


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

    # How training tiles are selected each epoch:
    #   "static"   -> (default) build one fixed, pre-filtered tile set at startup
    #                 using min_foreground_fraction/keep_empty_probability. Every
    #                 epoch iterates that same set. Simple and fully reproducible.
    #   "weighted" -> keep ALL tiles (no static filtering) but resample each epoch:
    #                 every positive tile (foreground >= min_foreground_fraction)
    #                 plus a fresh draw of `negative_ratio` x #positives negatives.
    #                 Gives the coverage of "all tiles" without the class imbalance,
    #                 and can bias the negative draw toward hard negatives the model
    #                 currently gets wrong (see hard_negative_* below). Set to
    #                 "weighted" to A/B against the static filter.
    tile_sampling: str = "static"

    # "weighted" only: negatives drawn per epoch as a multiple of the positive count.
    # 1.0 => balanced (one negative per positive). Ignored for "static".
    negative_ratio: float = 1.0

    # "weighted" only: fraction of the per-epoch negative draw taken from the
    # hardest negatives (highest recent training loss) rather than uniformly at
    # random. 0.0 => pure random negatives (a solid, feedback-free baseline);
    # 1.0 => all hardest. Ignored for "static".
    hard_negative_fraction: float = 0.0

    # "weighted" only: epochs of pure-random negative sampling before hard-negative
    # weighting activates, so difficulty estimates have signal first. Ignored for
    # "static" and when hard_negative_fraction == 0.
    hard_negative_warmup_epochs: int = 3

    # "weighted" only: curriculum-anneal negative_ratio from negative_ratio (start)
    # to negative_ratio_final over negative_ratio_anneal_epochs, so training starts
    # fiber-rich and ramps toward a more realistic (higher-negative) ratio without
    # the early-training collapse a high ratio can cause from the outset. Leave
    # negative_ratio_final null (default) to disable annealing entirely - the
    # sampler then uses a flat negative_ratio for the whole run, exactly as before.
    # Epoch length stays constant throughout (fixed at whichever of the two ratios
    # implies more negatives): epochs before the ramp completes pad the shortfall
    # by redrawing some positive tiles, rather than shrinking the batch count.
    negative_ratio_final: float | None = None
    negative_ratio_anneal_epochs: int = 0

    # Directory for precomputed, percentile-normalized full-image caches (.npy).
    # Avoids re-normalizing whole source images on every tile access. Shared across
    # runs/sweeps as long as source images don't change. Defaults to
    # "./cache/normalized_images" (relative to the working directory) when null.
    cache_dir: str | None = None

    # Number of channels returned by the dataset.
    # Use 1 for normal grayscale models.
    # Use 3 for MicroNet, where grayscale SEM images are copied into RGB-like channels.
    image_channels: int = 1

    # Final per-channel standardization applied to tiles AFTER percentile
    # normalization + augmentation, right before they become tensors:
    #   "minmax"   -> leave values in [0, 1] (default).
    #   "imagenet" -> subtract ImageNet mean and divide by ImageNet std, matching
    #                 what ImageNet/MicroNet-pretrained encoders were trained on.
    #                 Recommended whenever encoder_weights is "imagenet"/"micronet".
    #   "dataset"  -> subtract this dataset's OWN per-channel mean and divide by its
    #                 std (from norm_mean/norm_std below). Intended for FROM-SCRATCH
    #                 training (encoder_weights: null), where there is no pretrained
    #                 input distribution to match. Compute the stats with
    #                 `python -m fiberseg.tools.compute_dataset_stats --config <cfg> --write`.
    # Must match at train and inference time - predict_tiles.py reads the same
    # value so tiles are standardized identically. See dataset._apply_channel_norm.
    image_normalization: str = "minmax"

    # Per-channel mean/std for image_normalization: "dataset". Length must equal
    # image_channels. Left null otherwise; the compute_dataset_stats tool fills them
    # in from the TRAINING split only (never val/test, to avoid leakage).
    norm_mean: list[float] | None = None
    norm_std: list[float] | None = None


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

    # Soft-clDice topology term. When > 0, an auxiliary connectivity loss
    # (cldice_weight * (1 - soft_clDice)) is ADDED to whichever `name` loss is
    # selected. clDice rewards preserving the skeleton/centerline of thin tubular
    # structures, so it directly penalizes broken fibers - failures that region
    # losses (dice/tversky) barely notice because a severed fiber costs few pixels.
    # cldice_iters controls the soft-skeleton depth (higher = thicker fibers). 0
    # disables the term entirely (default), preserving prior behavior.
    cldice_weight: float = 0.0
    cldice_iters: int = 5

    # Focal BCE term. When > 0, an auxiliary per-pixel focal binary cross-entropy
    # loss (focal_bce_weight * FocalBCE) is ADDED to whichever `name` loss is
    # selected - independent of it, the same way cldice_weight is. Region losses
    # (dice/tversky) go near-zero-gradient on tiles the model already predicts
    # correctly (including confidently-correct empty tiles), so they give a weak
    # signal against confident false positives on hard-negative tiles. Focal BCE
    # gives a crisp per-pixel gradient there: focal_bce_gamma down-weights
    # already-easy/correct pixels so the loss concentrates on hard/wrong ones, and
    # focal_bce_alpha reweights the positive (fiber) vs negative (background)
    # pixel class. 0.0 disables the term entirely (default), preserving prior
    # behavior.
    focal_bce_weight: float = 0.0
    focal_bce_alpha: float = 0.25
    focal_bce_gamma: float = 2.0


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
class InferenceConfig:
    # How overlapping tile predictions are blended into the full-image probability
    # map in predict_tiles.predict_prob:
    #   "gaussian" -> weight each tile by a 2D Gaussian centered on the tile, so
    #                 unreliable tile-border pixels count far less than tile
    #                 centers (nnU-Net-style; default, reduces seam artifacts).
    #   "uniform"  -> plain average of overlapping tiles (previous behavior).
    tile_blend: str = "gaussian"

    # Reflect-pad partial edge tiles instead of zero-padding them, so the model
    # never sees an artificial black border at the image edge.
    reflect_pad: bool = True

    # Test-time augmentation: average sigmoid outputs over the 8 dihedral
    # (flip/rot90) variants of each tile. ~8x inference cost, typically +1-2 dice,
    # zero training cost. Off by default. The variants are generated and inverted
    # on the GPU (see predict_tiles._TTA_TRANSFORMS_TORCH), so enabling TTA costs
    # extra compute but no extra host<->device transfers.
    tta: bool = False

    # Tiles pushed through the model per forward pass. A large image is ~650 tiles
    # at patch_size 512 / stride 256; batching them amortizes per-launch overhead
    # and the GPU sync that a batch-of-1 loop pays per tile. Raise it if the GPU is
    # underused, lower it if inference OOMs (memory scales with batch_size x
    # patch_size^2, and TTA does not add to it - the variants run sequentially).
    #
    # Note this is not bit-for-bit neutral on an Ampere+ GPU: cuDNN selects different
    # convolution algorithms per batch shape, and with TF32 on (the default) that
    # moves individual probabilities by up to ~5e-2. Measured effect on the binarized
    # mask is nil (IoU 1.000000 between batch_size 1 and 8), and forcing
    # torch.backends.cudnn.allow_tf32 = False makes it exactly reproducible at ~28%
    # slower - worth doing only if you need bitwise-identical probability maps.
    batch_size: int = 8

    # How predict_tiles.predict_mask binarizes the probability map:
    #   "fixed"      -> a single cutoff, train.threshold (previous/default behavior).
    #   "hysteresis" -> Canny-style two-threshold hysteresis (hysteresis_low/_high):
    #                   pixels >= hysteresis_high seed confident fibre cores, pixels
    #                   >= hysteresis_low are kept only where connected to a seed, so
    #                   thin low-confidence fibre continuations survive instead of
    #                   being severed by a single hard threshold. See
    #                   tools/fiber_gap_repair.py's hysteresis_threshold_mask.
    #   "ridge"      -> run a Hessian ridge/vesselness filter (Frangi or Sato) over the
    #                   probability map FIRST, then threshold that response instead of
    #                   the raw probabilities. Because the filter scores how ridge-like
    #                   a neighbourhood is rather than how bright one pixel is, it
    #                   bridges along-fibre gaps and suppresses blobby false positives,
    #                   i.e. it builds a less-fragmented mask up front rather than
    #                   repairing a fragmented one afterwards (see
    #                   tools/fiber_gap_repair.py's enhance_ridges). Frangi's response
    #                   is already in [0, 1]; Sato's is not, so ridge_threshold means
    #                   different things between them - tune it per method.
    threshold_mode: str = "fixed"
    hysteresis_low: float = 0.3
    hysteresis_high: float = 0.7

    # "ridge" mode only. ridge_device selects the enhance_ridges backend: "cuda",
    # "cpu", or null for "cuda when available". The GPU path is the same filter
    # (validated in tests/test_gpu_ridges.py) but ~50x faster - a 46 MP Frangi is
    # ~60 s on CPU versus ~1 s on GPU, which is what makes this mode usable over a
    # whole dataset rather than a one-off experiment.
    ridge_method: str = "frangi"
    ridge_sigmas: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    ridge_threshold: float = 0.5
    ridge_device: str | None = None


@dataclass
class TrainConfig:
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32-true"
    threshold: float = 0.5

    # Seed for weight init, augmentation and dataloader shuffling. Deliberately separate
    # from data.seed, which fixes the train/val/test split: sweeping train.seed repeats
    # a run on the SAME split, which is how to measure run-to-run noise before trusting
    # a small metric difference between two configs. null falls back to data.seed
    # (the historical behaviour, so existing runs reproduce unchanged).
    seed: int | None = None

    # Metric that drives checkpoint selection, early stopping, and (for
    # reduce_on_plateau) LR scheduling. Any metric logged by the LightningModule
    # works. Default is "val/soft_tversky" (recall-weighted, threshold-free):
    # computed from continuous sigmoid probabilities rather than predictions
    # binarized at `threshold`, so it tracks the same quantity the loss
    # optimizes and isn't coupled to a threshold value that (per
    # tools/tune_threshold.py) is meant to be calibrated separately, after
    # training. "val/tversky"/"val/dice"/"val/iou" (hard, thresholded at
    # train.threshold) remain available and are still logged every epoch for
    # deployment-realistic reporting - just not used to drive training control
    # by default. monitor_mode is "max" for quality metrics, "min" for losses.
    monitor_metric: str = "val/soft_tversky"
    monitor_mode: str = "max"

    # Stochastic Weight Averaging (lightning StochasticWeightAveraging callback).
    # Averages weights over the tail of training for a small, cheap generalization
    # gain - especially useful on small datasets. swa_lr is the constant LR SWA
    # anneals to once it kicks in (at 75% of max_epochs). Note: SWA runs one extra
    # epoch at the end to recompute BatchNorm statistics on the averaged weights.
    swa: bool = False
    swa_lr: float = 1e-4

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
    inference: InferenceConfig = field(default_factory=InferenceConfig)
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

    _validate_normalization(data_cfg)
    _validate_sampling(data_cfg)
    _validate_loss((raw.get("train", {}) or {}).get("loss", {}) or {})
    _validate_inference(raw.get("inference", {}) or {})

    return AppConfig(
        data=_dataclass_from_dict(DataConfig, data_cfg, context="data"),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {}), context="model"),
        train=_dataclass_from_dict(TrainConfig, raw.get("train", {}), context="train"),
        inference=_dataclass_from_dict(
            InferenceConfig, raw.get("inference", {}), context="inference"
        ),
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