# Configuration guide

Every training / inference run is driven by a single YAML config, loaded by
[`fiberseg/config.py`](fiberseg/config.py) into nested dataclasses. This document lists
**every** parameter, its type, default, allowed values, and guidance for choosing a
setup. It is the human-readable companion to the dataclasses in `config.py` (the source
of truth) — if the two ever disagree, `config.py` wins.

- **Related docs:** [IMPROVEMENTS.md](IMPROVEMENTS.md) explains the *why* behind the
  performance options; [CLAUDE.md](CLAUDE.md) describes the code architecture.
- **Ready-made recipes:** see [`configs/`](configs/) — `example.yaml` (annotated
  starting point) and `micronet/unet_resnet50_improved_full.yaml` (recommended recipe).

---

## 1. How configs work

- The YAML has top-level **sections**: `data`, `model`, `train`, `inference`, `mlflow`,
  `logging`, `augmentations`, `sweep`, `foreground_filter_sweep`. Only `data` (with
  `images_dir` + `masks_dir`) is strictly required; every other section and field falls
  back to the defaults below.
- **Strict key checking.** An unknown key inside a known section, or an unknown
  top-level section, raises a `ValueError` at load — a typo fails loudly rather than
  being silently ignored. (So `log_every_pn_steps` would error; the correct key is
  `log_every_n_steps`.)
- **Validation at load:** `data.split` fractions must sum to 1; `image_normalization`
  must be a known mode; `norm_mean`/`norm_std` length must equal `image_channels`;
  `tile_sampling` must be known; `negative_ratio >= 0`; `hard_negative_fraction` in
  `[0,1]`. Some checks (e.g. missing `dataset` stats, MicroNet needing 3 channels) fire
  later, at training time, with a message pointing at the fix.
- **Nulls.** In YAML, `null` (or leaving a key out) means "use the default". Several
  fields explicitly accept `null` to mean "off" (e.g. `stride`, `encoder_weights`,
  `model_export_path`).

Run a config with:
```powershell
python -m fiberseg.train --config configs/your_config.yaml
```

---

## 2. Quick-start: choosing a setup

Pick the row that matches your situation, then use that column's settings as a base.

| You want… | `model.encoder_weights` | `data.image_channels` | `data.image_normalization` | `train.encoder_lr_ratio` |
|---|---|---|---|---|
| **MicroNet transfer** (recommended for SEM) | `micronet` | `3` | `imagenet` | `0.1` |
| **ImageNet transfer** | `imagenet` | `3` | `imagenet` | `0.1` |
| **From scratch** (no pretraining) | `null` | `1` | `dataset` (or `minmax`) | `null` |

Then choose the **loss** by what you care about most:

| Priority | `train.loss.name` | Extra |
|---|---|---|
| Balanced overlap | `bce_dice` | — |
| **Don't miss sparse fibers** (recall) | `bce_focal_tversky` | `tversky_beta` > `tversky_alpha` (e.g. 0.7 / 0.3) |
| **Keep thin fibers connected** (topology) | `bce_focal_tversky` | add `cldice_weight: 0.5` |

And the **tile strategy**:

| Situation | `data.tile_sampling` |
|---|---|
| Simple, reproducible baseline | `static` (default) |
| Full coverage + hard-negative mining (A/B once baseline is stable) | `weighted` |

Cheap post-training wins that need **no retrain**: run
`python -m fiberseg.tools.tune_threshold` to set `train.threshold`, and turn on
`inference.tta` for final predictions. See [IMPROVEMENTS.md](IMPROVEMENTS.md).

---

## 3. `data` — dataset, tiling, sampling, normalization

### Core

| Parameter | Type | Default | Values / range | Notes |
|---|---|---|---|---|
| `images_dir` | str | **required** | path | Folder of input images. |
| `masks_dir` | str | **required** | path | Folder of ground-truth masks (values > 0 = fiber). |
| `image_glob` | str | `"*.tif"` | glob | Which files under `images_dir` are inputs. Files ending `_mask` are always excluded. |
| `mask_pattern` | str | `"{stem}_mask.tif"` | template | Mask filename from image, via `{stem}`/`{suffix}`/`{name}` placeholders. |
| `patch_size` | int or `[h, w]` | `512` | > 0 | Tile size fed to the model. |
| `stride` | int or `[h, w]` or null | `null` | > 0 | Tile step. `null` = no overlap (stride = patch_size). Smaller = more overlap, more tiles, slower. Typical: half the patch (e.g. 256 for 512). |
| `split` | map | `{train: 0.7, val: 0.15, test: 0.15}` | fractions summing to 1 | Split is **by image**, not by tile (no tile leakage across splits). |
| `seed` | int | `42` | any | Controls the image split and tile filtering — change to get a different split. |
| `num_workers` | int | `4` | ≥ 0 | DataLoader worker processes. 0 = load in main process (simplest for debugging). |
| `batch_size` | int | `4` | ≥ 1 | Tiles per step. Raise until GPU memory is full; lower if you hit OOM. |
| `cache_dir` | str or null | `null` | path | Where percentile-normalized `.npy` caches live. `null` → `./cache/normalized_images`. |
| `image_channels` | int | `1` | `1` or `3` | `1` = grayscale; `3` = replicate grayscale to RGB (**required** for `imagenet`/`micronet` encoders). |

### Static tile filtering (used when `tile_sampling: static`)

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `min_foreground_fraction` | float | `0.0` | `[0, 1]` | A tile is "empty" if its fiber-pixel fraction is below this. `0.0` = keep everything. Sparse fiber masks: try `0.001`. |
| `keep_empty_probability` | float | `1.0` | `[0, 1]` | Probability of keeping an empty tile. `1.0` = keep all; lower to down-sample background (e.g. `0.2` keeps ~20% of empties). Only meaningful with `min_foreground_fraction > 0`. |

### Weighted sampling (used when `tile_sampling: weighted`) — see [IMPROVEMENTS.md §8b](IMPROVEMENTS.md)

| Parameter | Type | Default | Values / range | Notes |
|---|---|---|---|---|
| `tile_sampling` | str | `"static"` | `static`, `weighted` | `weighted` keeps all tiles and resamples every epoch (all positives + fresh negatives). |
| `negative_ratio` | float | `1.0` | ≥ 0 | `weighted` only: negatives drawn per positive per epoch. `1.0` = balanced. |
| `hard_negative_fraction` | float | `0.0` | `[0, 1]` | `weighted` only: share of the negative draw taken from the hardest negatives (highest recent loss). `0.0` = pure random (feedback-free baseline). |
| `hard_negative_warmup_epochs` | int | `3` | ≥ 0 | `weighted` only: random-only epochs before hard mining activates. |

### Input standardization — see [IMPROVEMENTS.md §1](IMPROVEMENTS.md)

| Parameter | Type | Default | Values | Notes |
|---|---|---|---|---|
| `image_normalization` | str | `"minmax"` | `minmax`, `imagenet`, `dataset` | Final per-channel step after percentile normalization. **Must match at train and inference.** |
| `norm_mean` | list[float] or null | `null` | length = `image_channels` | For `dataset` mode. Fill via `python -m fiberseg.tools.compute_dataset_stats --config <cfg> --write`. |
| `norm_std` | list[float] or null | `null` | length = `image_channels` | As above. |

**Normalization modes:**
- `minmax` — leave values in `[0,1]`. Fine for from-scratch training.
- `imagenet` — subtract ImageNet mean / divide by std. Use with `imagenet`/`micronet` encoders.
- `dataset` — use your own training-split mean/std. For from-scratch training only; never leaks val/test (stats computed on train split).

---

## 4. `model` — architecture and encoder

| Parameter | Type | Default | Values | Notes |
|---|---|---|---|---|
| `architecture` | str | `"Unet"` | `Unet`, `UnetPlusPlus`, `MAnet`, `Linknet`, `FPN`, `PSPNet`, `PAN`, `DeepLabV3`, `DeepLabV3Plus` | Any `segmentation_models_pytorch` architecture, looked up by name. `Unet`/`UnetPlusPlus` are the usual choices for fibers. |
| `encoder_name` | str | `"resnet34"` | 113 options (see below) | Backbone. Bigger = more capacity + compute. |
| `encoder_weights` | str or null | `null` | `null`, `imagenet`, `micronet` | Pretraining source. `null` = from scratch. `micronet` = NASA microscopy weights (needs the optional `pretrained_microscopy_models` package and `image_channels: 3`). |
| `in_channels` | int | `1` | `1` or `3` | Must equal `data.image_channels`. `micronet` requires `3`. |
| `classes` | int | `1` | `1` | Binary segmentation → keep `1` (background implicit). |

**Common `encoder_name` values** (113 total from smp): `resnet18`, `resnet34`, `resnet50`,
`resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `se_resnet50`,
`efficientnet-b0` … `efficientnet-b7`, `mobilenet_v2`, `timm-efficientnet-b3`, and more.
Larger backbones (resnet50+) benefit most from pretrained weights; resnet18/34 train
fast from scratch. For `micronet`, the encoder must be one the NASA package provides
weights for (e.g. `resnet50`, `se_resnet50`).

> **Contract:** every model outputs **raw logits** — sigmoid is applied only inside the
> loss/metrics/inference. Don't add an output activation.

---

## 5. `train` — optimization, schedule, checkpoints

### Core

| Parameter | Type | Default | Values / range | Notes |
|---|---|---|---|---|
| `max_epochs` | int | `50` | ≥ 1 | Upper bound; early stopping usually ends sooner. |
| `learning_rate` | float | `1e-4` | > 0 | Base LR (AdamW). Pretrained: `1e-4`–`2e-4`; from scratch can go higher. |
| `weight_decay` | float | `1e-4` | ≥ 0 | AdamW weight decay. |
| `encoder_lr_ratio` | float or null | `null` | > 0 | Encoder LR = `learning_rate × ratio`. Use `0.1` with pretrained encoders; `null` (single LR) for from scratch. |
| `accelerator` | str | `"auto"` | `auto`, `gpu`, `cpu` | `auto` picks GPU if available. |
| `devices` | str or int | `"auto"` | `auto`, int, list | Number/IDs of devices. |
| `precision` | str | `"32-true"` | `32-true`, `16-mixed`, `bf16-mixed`, `64-true` | On GPU, `32-true` is auto-upgraded to `16-mixed` for speed. `bf16-mixed` on Ampere+ GPUs. |
| `threshold` | float | `0.5` | `[0, 1]` | Probability cut for a pixel = fiber. Tune post-training with `tools/tune_threshold.py`. |
| `matmul_precision` | str or null | `"medium"` | `medium`, `high`, `highest`, null | Torch float32 matmul precision (`medium` = fastest with TF32). |
| `log_every_n_steps` | int | `10` | ≥ 1 | MLflow logging cadence. |
| `val_check_interval` | float | `1.0` | `(0, 1]` or int | `1.0` = validate once per epoch; `0.5` = twice; int = every N steps. |
| `deterministic` | bool | `false` | bool | Fully deterministic (slower). |
| `resume_from_checkpoint` | str or null | `null` | path | Restore model+optimizer+epoch from a `.ckpt`. Also settable via `--resume`. |

### Model selection / early stopping — see [IMPROVEMENTS.md §10](IMPROVEMENTS.md)

| Parameter | Type | Default | Values | Notes |
|---|---|---|---|---|
| `monitor_metric` | str | `"val/tversky"` | `val/tversky`, `val/dice`, `val/iou`, `val/precision`, `val/recall`, `val/f2`, `val/loss` | Metric driving checkpoint selection, early stopping, and the plateau scheduler. Match your real target. |
| `monitor_mode` | str | `"max"` | `max`, `min` | `max` for quality metrics, `min` for `val/loss`. |

### Stochastic Weight Averaging — see [IMPROVEMENTS.md §7](IMPROVEMENTS.md)

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `swa` | bool | `false` | bool | Average weights over the last 25% of training. One extra epoch to recompute BatchNorm. |
| `swa_lr` | float | `1e-4` | > 0 | Constant LR SWA anneals to. |

### Diagnostics / export

| Parameter | Type | Default | Values | Notes |
|---|---|---|---|---|
| `profiler` | str or null | `null` | `null`, `simple`, `advanced`, `pytorch` | Bottleneck profiling. Also settable via `--profile`. |
| `limit_train_batches` | float/int or null | `null` | fraction or count | Cap train batches/epoch for quick runs. |
| `limit_val_batches` | float/int or null | `null` | fraction or count | Cap val batches/epoch. |
| `model_export_path` | str or null | `null` | path | If set, export best checkpoint to TorchScript here after training. |

### `train.loss` — loss function

| Parameter | Type | Default | Values / range | Notes |
|---|---|---|---|---|
| `name` | str | `"bce_dice"` | `bce`, `dice`, `bce_dice`, `tversky`, `bce_tversky`, `focal_tversky`, `bce_focal_tversky` | Base loss (see table below). |
| `tversky_alpha` | float | `0.3` | `[0, 1]` | Penalty weight on false **positives**. |
| `tversky_beta` | float | `0.7` | `[0, 1]` | Penalty weight on false **negatives** (missed fibers). `beta > alpha` → recall-tilted. Typically `alpha + beta ≈ 1`. |
| `focal_gamma` | float | `0.75` | > 0 | Focal exponent for `focal_tversky` variants; > 1 focuses harder on difficult regions, < 1 softer. |
| `cldice_weight` | float | `0.0` | ≥ 0 | If > 0, adds `weight × (1 − soft_clDice)` on top of the base loss (topology term for thin fibers). Try `0.3`–`1.0`. |
| `cldice_iters` | int | `5` | ≥ 1 | Soft-skeleton depth; higher tolerates thicker fibers. |

**Loss choices:**

| `name` | What it optimizes | Use when |
|---|---|---|
| `bce` | Per-pixel classification | Balanced masks; rarely best alone for sparse fibers. |
| `dice` | Region overlap | Overlap-focused, imbalance-robust. |
| `bce_dice` | Both | Solid general default. |
| `tversky` | Tunable FP/FN trade-off | You need to weight missed vs false fibers. |
| `bce_tversky` | BCE + Tversky | Adds pixel signal to Tversky. |
| `focal_tversky` | Tversky, focused on hard regions | Sparse masks with hard boundaries. |
| `bce_focal_tversky` | BCE + focal Tversky | **Recommended for sparse fibers.** |

### `train.scheduler` — learning-rate schedule

| Parameter | Type | Default | Values | Notes |
|---|---|---|---|---|
| `name` | str | `"reduce_on_plateau"` | `reduce_on_plateau`, `cosine`, `none`/`off`/`false` | Plateau watches `monitor_metric`; cosine anneals over `max_epochs`. |
| `patience` | int | `5` | ≥ 1 | (plateau) epochs without improvement before reducing LR. |
| `factor` | float | `0.5` | `(0, 1)` | (plateau) LR multiplier on reduction. |
| `min_lr` | float | `1e-6` | ≥ 0 | Floor LR (plateau) / `eta_min` (cosine). |

### `train.early_stopping`

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `patience` | int | `15` | Epochs without `monitor_metric` improvement before stopping. Only attached when `max_epochs >= 10`. |
| `min_delta` | float | `0.001` | Minimum improvement that counts. |

---

## 6. `inference` — tiled prediction quality — see [IMPROVEMENTS.md §3–4](IMPROVEMENTS.md)

| Parameter | Type | Default | Values | Notes |
|---|---|---|---|---|
| `tile_blend` | str | `"gaussian"` | `gaussian`, `uniform` | How overlapping tiles are merged. `gaussian` down-weights unreliable tile borders (fewer seams). |
| `reflect_pad` | bool | `true` | bool | Reflect-pad edge tiles instead of zero-padding (no artificial black border). |
| `tta` | bool | `false` | bool | 8× dihedral test-time augmentation. ~8× inference cost, no retrain, typically +1–2 dice. Turn on for final predictions. |

---

## 7. `mlflow`, `logging`

### `mlflow`

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `tracking_uri` | str | `"http://127.0.0.1:5000"` | Start the server (`start_mlflow.bat`) before training. |
| `experiment_name` | str | `"fiber-sem-segmentation"` | MLflow experiment to log under. |
| `run_name` | str or null | `null` | `null` → auto-generated from architecture/encoder/patch/LR. |

### `logging`

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `log_prediction_images` | bool | `true` | Log periodic + best-model sample predictions as MLflow image artifacts. |
| `prediction_max_images` | int | `4` | How many best-model sample images to log. |
| `prediction_artifact_dir` | str | `"predictions/best_model_test"` | Artifact subfolder for logged images. |

---

## 8. `augmentations` — training-time transforms

A mapping of split name → list of [Albumentations](https://albumentations.ai/docs/) transforms.
Each item is `{name: <TransformName>, <param>: <value>, ...}`. Geometric transforms are
applied to image and mask together; photometric ones to the image only.

```yaml
augmentations:
  train:
    - {name: HorizontalFlip, p: 0.5}
    - {name: VerticalFlip, p: 0.5}
    - {name: RandomRotate90, p: 0.5}
    - name: Affine
      scale: [0.75, 1.35]        # fiber width varies with SEM magnification
      translate_percent: [-0.05, 0.05]
      rotate: [-15, 15]
      p: 0.5
    - {name: RandomBrightnessContrast, brightness_limit: 0.15, contrast_limit: 0.15, p: 0.4}
    - {name: GaussNoise, std_range: [0.02, 0.08], p: 0.3}   # see caveat below
    - {name: GaussianBlur, blur_limit: [3, 5], p: 0.2}
  val: []
  test: []
```

- **Any Albumentations transform** is accepted by class name; unknown names raise at load.
- **`GaussNoise` caveat:** this pipeline normalizes to `[0,1]` before augmenting, so you
  **must** set `std_range` explicitly (e.g. `[0.02, 0.08]`). The Albumentations default is
  calibrated for `[0,255]` and will saturate the image with noise — the code rejects
  `GaussNoise` without an explicit `std_range`.
- **Avoid elastic/grid-distortion** transforms — they corrupt thin-mask geometry.
- Keep `val`/`test` empty (`[]`) for deterministic evaluation.

---

## 9. `sweep` — grid search (optional)

A mapping of **dotted config keys → list of values**. `python -m fiberseg.train` expands the
cartesian product into one run per combination automatically.

```yaml
sweep:
  model.architecture: ["Unet", "UnetPlusPlus"]
  model.encoder_name: ["resnet34", "resnet50"]
  train.learning_rate: [0.0001, 0.0002]
```
The example above launches 2 × 2 × 2 = 8 runs. See CLAUDE.md ("two sweep mechanisms")
before using `python -m fiberseg.sweep` — prefer the built-in expansion in `train`.

---

## 10. Minimal and full examples

### Minimal (relies on all defaults)
```yaml
data:
  images_dir: "data/images"
  masks_dir: "data/masks"
```

### Recommended MicroNet recipe (abridged)
```yaml
data:
  images_dir: "data/images"
  masks_dir: "data/masks"
  image_channels: 3
  image_normalization: "imagenet"
  patch_size: 512
  stride: 256
  batch_size: 8
  min_foreground_fraction: 0.001
  keep_empty_probability: 0.2

model:
  architecture: "Unet"
  encoder_name: "resnet50"
  encoder_weights: "micronet"
  in_channels: 3
  classes: 1

train:
  max_epochs: 400
  learning_rate: 0.0002
  encoder_lr_ratio: 0.1
  monitor_metric: "val/tversky"
  swa: true
  loss:
    name: "bce_focal_tversky"
    tversky_alpha: 0.3
    tversky_beta: 0.7
    cldice_weight: 0.5
  scheduler:
    name: "reduce_on_plateau"
    patience: 4
  early_stopping:
    patience: 15

inference:
  tile_blend: "gaussian"
  reflect_pad: true
  tta: false
```

A complete, runnable version of this recipe lives at
[`configs/micronet/unet_resnet50_improved_full.yaml`](configs/micronet/unet_resnet50_improved_full.yaml).
