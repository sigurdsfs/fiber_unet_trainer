# Fiber U-Net Trainer

This repository trains and evaluates binary segmentation models for SEM fiber images using PyTorch, PyTorch Lightning, segmentation-models-pytorch, and MLflow. It is designed for local experimentation and for processing large microscopy images by tiling them into patches during training and prediction.

## What this project does

- Trains a segmentation model for fiber-vs-background masks
- Uses tiled sampling for very large grayscale images
- Supports configurable augmentations, losses, and learning-rate scheduling
- Logs experiments locally with MLflow
- Can export a trained checkpoint to a TorchScript model

## Repository layout

- [fiberseg](fiberseg) — core training, dataset, model, and prediction code
  - [fiberseg/train.py](fiberseg/train.py) — entry point for single-model training
  - [fiberseg/sweep.py](fiberseg/sweep.py) — grid-search / sweep runner
  - [fiberseg/predict_tiles.py](fiberseg/predict_tiles.py) — tiled inference on large images
  - [fiberseg/config.py](fiberseg/config.py) — config loading and validation
  - [fiberseg/dataset.py](fiberseg/dataset.py) — dataset and tile sampling
  - [fiberseg/lit_module.py](fiberseg/lit_module.py) — Lightning module
- [configs](configs) — example YAML configuration files
- [tests](tests) — lightweight regression and environment checks
- [exported_models](exported_models) — example exported model artifacts
- [mlruns](mlruns) — local MLflow tracking store

## Expected data layout

The default config expects image and mask folders similar to this:

```text
data/
  images/
    sample01.tif
    sample02.tif
  masks/
    sample01_mask.tif
    sample02_mask.tif
```

Important notes:

- Masks should be binary or label images where values greater than zero mean fiber and zero means background.
- Training and validation are split by image, not by tile, to avoid leakage.
- If your filenames differ, update the mask pattern in the YAML config.

Example patterns:

```yaml
data:
  mask_pattern: "{stem}_mask.tif"
  mask_pattern: "{stem}.tif"
  mask_pattern: "mask_{stem}.tif"
```

## Environment setup

A typical Conda workflow on Windows:

```powershell
conda create -n cnn_test python=3.11 -y
conda activate cnn_test
pip install -r requirements.txt
```

If you want editable installs and local development tooling:

```powershell
pip install -e .[dev]
```

## MLflow setup

Start the local MLflow UI before training:

```powershell
start_mlflow.bat
```

Then open:

```text
http://127.0.0.1:5000
```

## Train a single model

1. Edit [configs/example.yaml](configs/example.yaml) and point the data paths to your dataset.
2. Run:

```powershell
python -m fiberseg.train --config configs/example.yaml
```

This will:

- load the YAML config
- create a Lightning trainer
- train the model
- log metrics and artifacts to MLflow
- save a best checkpoint and a last checkpoint

## Run a sweep

The sweep runner uses the YAML config’s sweep section to explore combinations of hyperparameters.

Example:

```yaml
sweep:
  data.patch_size: [256, 512]
  data.stride: [256, 512]
  train.learning_rate: [0.0001, 0.0003]
  model.encoder_name: ["resnet18", "resnet34"]
```

Run:

```powershell
python -m fiberseg.sweep --config configs/example.yaml
```

## Run prediction on a large image

After training, use the best checkpoint and run tiled inference:

```powershell
python -m fiberseg.predict_tiles ^
  --config configs/example.yaml ^
  --checkpoint path/to/best.ckpt ^
  --image data/images/sample01.tif ^
  --out prediction_mask.tif
```

## Export a trained model

To export a checkpoint to TorchScript:

```powershell
python export_model.py
```

The script will help you pick a checkpoint, locate the matching config, and export the model to the [exported_models](exported_models) folder.

## Useful training settings

For sparse fiber masks, these values often help:

```yaml
data:
  min_foreground_fraction: 0.001
  keep_empty_probability: 0.2
```

Useful losses include:

```yaml
train:
  loss: "bce_dice"
  loss: "focal_tversky"
  loss: "bce_focal_tversky"
```

The default config already includes adaptive learning-rate scheduling and Tversky-based metrics.

## Development and testing

Run the checks locally:

```powershell
pytest
ruff check .
```

## Notes for future agents and users

- Start from [configs/example.yaml](configs/example.yaml) when adapting the pipeline.
- The most important runtime entry points are [fiberseg/train.py](fiberseg/train.py), [fiberseg/sweep.py](fiberseg/sweep.py), and [fiberseg/predict_tiles.py](fiberseg/predict_tiles.py).
- Config loading and validation happen in [fiberseg/config.py](fiberseg/config.py).
- The training logic and metrics are implemented in [fiberseg/lit_module.py](fiberseg/lit_module.py).
- MLflow tracking is local by default and writes to [mlruns](mlruns).
