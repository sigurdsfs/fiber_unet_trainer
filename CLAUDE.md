# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This directory (`fiber_unet_trainer_v2 - Sigurd`) contains a single Python project in the
[fiber_unet_trainer/](fiber_unet_trainer/) subfolder — that is the working directory for all
commands below. It trains and evaluates binary segmentation models for SEM fiber images using
PyTorch, PyTorch Lightning, `segmentation-models-pytorch`, and MLflow, and tiles large microscopy
images into patches for both training and inference.

## Commands

All commands assume `cd fiber_unet_trainer` and an activated environment (see below).

```powershell
# Environment
conda create -n cnn_test python=3.11 -y
conda activate cnn_test
pip install -r requirements.txt
pip install -e .[dev]        # editable install + pytest/ruff/black/pre-commit

# Local MLflow UI (start before training so runs are logged/visible)
start_mlflow.bat             # serves at http://127.0.0.1:5000

# Train a single model
python -m fiberseg.train --config configs/example.yaml

# Run a grid sweep (see "Two sweep mechanisms" below before using this)
python -m fiberseg.sweep --config configs/example.yaml

# Tiled inference on one large image using a trained checkpoint
python -m fiberseg.predict_tiles --config configs/example.yaml --checkpoint path/to/best.ckpt --image data/images/sample01.tif --out prediction_mask.tif

# Tiled inference on every image under the config's data.images_dir/data.image_glob
python -m fiberseg.predict_all --config configs/example.yaml --checkpoint path/to/best.ckpt --out-dir predictions/

# Export a checkpoint to TorchScript (interactive picker over mlruns/lightning_logs)
python export_model.py

# Tests and lint
pytest                       # runs tests/ (see pyproject.toml testpaths)
pytest tests/test_config_validation.py::test_load_config_rejects_invalid_split_fractions  # single test
ruff check .
```

Note: [tests/test_micronet_forward.py](fiber_unet_trainer/tests/test_micronet_forward.py) and
[tests/test_gpu_setup.py](fiber_unet_trainer/tests/test_gpu_setup.py)-adjacent scripts like it are
not always plain pytest cases — `test_micronet_forward.py` defines only a `main()` guarded by
`if __name__ == "__main__"` (no `test_*` function), requires a real MicroNet config/checkpoint, and
is meant to be run directly with `python tests/test_micronet_forward.py`, not collected by pytest.

## Architecture

### Config-driven pipeline

Everything is driven by one YAML file (see [configs/example.yaml](fiber_unet_trainer/configs/example.yaml)).
[fiberseg/config.py](fiber_unet_trainer/fiberseg/config.py) loads it into a nested dataclass
`AppConfig` (`data`, `model`, `train`, `mlflow`, `logging`, plus raw `augmentations`/`sweep` dicts).
`load_config` validates `data.images_dir`/`masks_dir` are present and that `data.split` fractions
sum to 1. Unknown YAML keys under a section are silently dropped (`_dataclass_from_dict` filters to
known dataclass fields) rather than erroring — keep this in mind when a config value seems to have
no effect. `to_dict()` round-trips an `AppConfig` back to plain dict/list for logging the resolved
config as an MLflow artifact.

### Data flow: pairing → tiling → dataset

[fiberseg/dataset.py](fiber_unet_trainer/fiberseg/dataset.py) does all data handling:
- `find_pairs()` globs `images_dir` for `image_glob`, derives each mask path via
  `mask_pattern.format(stem=..., suffix=..., name=...)`, and **splits by image (not by tile)** using
  a `random.Random(seed)` shuffle — this avoids leaking tiles from the same source image across
  train/val/test.
- `TiledSegmentationDataset` turns each image into a grid of overlapping tiles (`patch_size`/`stride`,
  each may be an int or `[h, w]`); the last row/column of tiles is snapped to the image edge so the
  full image is always covered. For the `train` split only, tiles can be filtered by
  `min_foreground_fraction`/`keep_empty_probability` to downsample empty background tiles in sparse
  fiber masks (filtered counts are tracked and logged to MLflow — see `_log_filtered_tile_stats` in
  train.py).
- Images/masks are cached with `@lru_cache(maxsize=16)` on path string (`_cached_read`) since the
  same source image is read for many tiles.
- `_normalize_image` does percentile (1st/99.5th) contrast normalization to `[0, 1]`, robust to
  bright outlier pixels typical of SEM images.
- `image_channels` controls whether tiles are returned as 1-channel grayscale or replicated to
  3 channels (required for ImageNet/MicroNet-pretrained encoders).
- `FiberDataModule` (LightningDataModule) wraps train/val/test datasets and builds
  per-split Albumentations transforms from the config's `augmentations` dict.

### Model creation and the raw-logits contract

[fiberseg/models.py](fiber_unet_trainer/fiberseg/models.py) `create_model()` dispatches on
`model.encoder_weights`:
- Normal case → any `segmentation_models_pytorch` architecture (`Unet`, `UnetPlusPlus`, `FPN`,
  `DeepLabV3Plus`, ...) looked up dynamically by name from `smp`, with `activation=None`.
- `encoder_weights: "micronet"` → NASA's `pretrained_microscopy_models` package (external, optional
  dependency, not in `requirements.txt` — install from GitHub if needed) via
  `create_segmentation_model`, then `force_raw_logits()` strips any built-in output activation
  (Sigmoid/Softmax/Activation wrapper) it may add.

**Every model returned by `create_model` must output raw logits, never sigmoid/softmax
probabilities** — sigmoid is applied only inside the loss/metrics ([fiberseg/lit_module.py](fiber_unet_trainer/fiberseg/lit_module.py))
and inference ([fiberseg/predict_tiles.py](fiber_unet_trainer/fiberseg/predict_tiles.py)) code. If you
add a new model backend, preserve this contract.

### Lightning module: losses, metrics, scheduling

[fiberseg/lit_module.py](fiber_unet_trainer/fiberseg/lit_module.py) `FiberSegmentationLitModule`
computes binary stats (dice/iou/precision/recall/tversky/f2) from thresholded logits, and selects a
loss by string name (`train.loss`): `bce`, `dice`, `bce_dice`, `tversky`, `bce_tversky`,
`focal_tversky`, `bce_focal_tversky` — the last two use a differentiable soft Tversky index and are
recommended for sparse fiber masks (`tversky_beta` > `tversky_alpha` penalizes missed fibers more).
`configure_optimizers` supports `scheduler: reduce_on_plateau` (monitors `val/tversky`, the default)
or `cosine`; both checkpoint-worthy metrics (`ModelCheckpoint`, `EarlyStopping` in train.py) also key
off `val/tversky`.

### Training entry point and the two sweep mechanisms

[fiberseg/train.py](fiber_unet_trainer/fiberseg/train.py) `run_training(cfg)` **auto-detects a
`sweep:` section in the loaded config and expands it into one training run per combination itself**
(`_expand_sweep_configs`, cartesian product of `sweep` dict values, dotted-key assignment via
`_set_nested`). This means `python -m fiberseg.train --config <cfg-with-sweep-section>` already runs
a full grid.

[fiberseg/sweep.py](fiber_unet_trainer/fiberseg/sweep.py) is a second, separate sweep runner: it
also cartesian-products `cfg.sweep`, sets one combination's values on a deep copy, wraps each run in
its own `mlflow.start_run(...)`, and then calls `run_training(cfg)` from train.py — but it does not
clear `cfg.sweep` on that copy first, so `run_training` will detect the sweep section is still
present and expand the *entire original grid again* inside each outer iteration. Read both files
before changing sweep behavior; don't assume `python -m fiberseg.sweep` runs the grid exactly once.

### Inference and export

[fiberseg/predict_tiles.py](fiber_unet_trainer/fiberseg/predict_tiles.py) reuses `_hw`,
`_normalize_image`, `_read_gray` from `dataset.py` to keep preprocessing identical between training
and inference. It tiles the whole image with overlap, averages overlapping sigmoid probabilities
into an accumulator, then thresholds with `train.threshold`. The checkpoint-loading
(`load_predictor`), per-image tiling/inference (`predict_mask`), and mask-writing (`save_mask`)
steps are factored into reusable functions so other scripts don't reimplement the tiling loop.

[fiberseg/predict_all.py](fiber_unet_trainer/fiberseg/predict_all.py) reuses those same
`predict_tiles` functions to run inference over every image under a config's
`data.images_dir`/`data.image_glob` (same discovery/exclusion rule as `find_pairs`, minus the
requirement for a matching mask), loading the checkpoint once and writing one output mask per
input image into `--out-dir`.

[export_model.py](fiber_unet_trainer/export_model.py) is an interactive CLI (top-level, not under
`fiberseg/`) that scans `mlruns/`, `lightning_logs/`, and `.` for `*.ckpt` files, lets you pick one,
locates the matching config, and exports to TorchScript under `exported_models/`.
[fiberseg/tools/](fiber_unet_trainer/fiberseg/tools/) has related standalone scripts:
`export_torchscript.py`, `inspect_checkpoint.py`, `preview_augmentations.py`.

### MLflow

All training runs log to a local MLflow tracking store at `mlflow.tracking_uri`
(`http://127.0.0.1:5000` by default, backed by [mlruns/](fiber_unet_trainer/mlruns/)) — start it with
`start_mlflow.bat` before training or metrics/artifacts calls will fail to connect. Prediction image
callbacks ([fiberseg/callbacks.py](fiber_unet_trainer/fiberseg/callbacks.py)) log periodic and
best-model sample predictions as MLflow image artifacts during training.

## Data layout expected by configs

```text
data/
  images/
    sample01.tif
  masks/
    sample01_mask.tif        # values > 0 = fiber, 0 = background
```

Filenames ending in `_mask` are excluded from the image glob (`find_pairs`) since they're treated as
ground-truth masks, not inputs. Mask filename matching is controlled by `data.mask_pattern`
(`{stem}`/`{suffix}`/`{name}` placeholders).
