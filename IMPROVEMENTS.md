# Model performance improvements

This document records the ML-quality changes made to the pipeline, why each one
helps, how to turn it on, and its compute cost. They are ordered by expected gain
per unit of compute. Everything is **opt-in via config** — existing configs keep
their previous behavior unless you set the new keys.

The recommended recipe that bundles all of these is
[configs/micronet/unet_resnet50_improved.yaml](configs/micronet/unet_resnet50_improved.yaml).

---

## Tier 1 — free wins (little or no extra training compute)

### 1. Input standardization (`data.image_normalization`)
The final per-tile standardization step, applied after percentile normalization.
Three modes:

- `"minmax"` (default, previous behavior) — leave values in `[0,1]`.
- `"imagenet"` — subtract ImageNet mean / divide by ImageNet std. Pretrained
  ImageNet/MicroNet encoders were trained on inputs standardized this way, so the
  old raw-`[0,1]` path forced their first layers to re-adapt. **Use this whenever
  `encoder_weights` is `"imagenet"`/`"micronet"`.** 3-channel uses per-RGB
  constants; 1-channel uses the ImageNet grayscale mean/std.
- `"dataset"` — subtract **this dataset's own** per-channel mean / divide by its std
  (`data.norm_mean`/`data.norm_std`). **For from-scratch training
  (`encoder_weights: null`)**, where there is no pretrained input distribution to
  match, so standardizing to your own data is the natural choice. Compute the stats
  over the **training split only** (no val/test leakage) with:
  ```powershell
  python -m fiberseg.tools.compute_dataset_stats --config <cfg> --write
  ```
  `--write` inserts `norm_mean`/`norm_std` and `image_normalization: dataset` into
  the config's `data:` block (comments preserved). Training errors early if `dataset`
  mode is set without stats, and warns if it's combined with a pretrained encoder.

**Applies at train and inference identically** — `dataset._apply_channel_norm` is
called in both `dataset.__getitem__` and `predict_tiles._make_model_input`, reading
the same mode (and the same `norm_mean`/`norm_std` for `dataset`). Never set them
inconsistently; a checkpoint trained with one mode must be predicted with the same
mode. **Cost:** none. **Requires retraining** to benefit (it changes the input dist).

> Note on pretrained encoders: use the encoder's *pretraining* stats (`imagenet`),
> not your dataset's — the pretrained filters are calibrated to that input space.
> `dataset` stats are for from-scratch training. (`pmm` also ships MicroNet-specific
> stats `[0.4723, 0.4599, 0.4468]` but doesn't use them by default; `imagenet`
> matches its documented path.)

### 2. Decision-threshold tuning (`tools/tune_threshold.py`)
The recall-weighted loss (`tversky_beta > tversky_alpha`) biases sigmoid outputs,
so 0.5 is rarely the dice-optimal cut. Sweep the threshold on the validation split
over full-resolution probability maps:

```powershell
python -m fiberseg.tools.tune_threshold --config <cfg> --checkpoint <best.ckpt> --metric dice
```

Then set the printed value as `train.threshold`. **Cost:** seconds–minutes, no
retrain. Measured a **+4.9 dice** gain (0.736 → 0.785) on the existing checkpoint.
Caveat: with only 4 val images the optimum is noisy — see Tier 3 item 9.

### 3. Gaussian tile blending + reflect padding (`inference.tile_blend`, `inference.reflect_pad`)
`predict_prob` now weights each overlapping tile by a 2D Gaussian centered on the
tile (unreliable tile-border pixels count far less than centers — the nnU-Net
sliding-window scheme) and reflect-pads partial edge tiles instead of zero-padding
(no artificial black border). Defaults: `tile_blend: "gaussian"`, `reflect_pad: true`.
Set `tile_blend: "uniform"` for the old averaging. **Cost:** none.

### 4. Test-time augmentation (`inference.tta`)
Average sigmoid outputs over the 8 dihedral (flip/rot90) views of each tile. The
transforms are exact inverses (unit-tested). Off by default; set `inference.tta:
true` for final predictions. **Cost:** ~8× inference only, no training cost,
typically +1–2 dice.

---

## Tier 2 — small training-cost changes

### 5. Differential encoder learning rate (`train.encoder_lr_ratio`)
Give the pretrained encoder its own param group at `learning_rate *
encoder_lr_ratio` (e.g. `0.1`) while the fresh decoder keeps the full LR — the
standard transfer-learning trick that protects pretrained features from noisy
early decoder gradients. The plumbing already existed in `lit_module`; the recipe
now sets `encoder_lr_ratio: 0.1`. **Cost:** none.

### 6. Soft-clDice topology loss (`train.loss.cldice_weight`, `train.loss.cldice_iters`)
Fibers are thin tubular structures; region losses (dice/tversky) barely penalize a
severed fiber because it costs few pixels, yet connectivity is what matters for
fiber counting/length. clDice compares the *skeletons* of prediction and target
(differentiable soft-skeletonization via min/max pooling) and rewards preserving
centerlines. When `cldice_weight > 0`, `cldice_weight * (1 - soft_clDice)` is added
to whichever base `loss.name` is selected — it composes with any base loss. Verified
that a broken fiber scores lower than a perfect one (0.91 vs 1.00). **Cost:** a few
extra pooling ops per step (small).

### 7. Stochastic Weight Averaging (`train.swa`, `train.swa_lr`)
Averages weights over the last 25% of training (Lightning's
`StochasticWeightAveraging`) for a cheap generalization gain, especially valuable
on small datasets. **Cost:** ~1 extra epoch at the end to recompute BatchNorm stats.

### 8. Richer, SEM-realistic augmentations
The improved config adds calibrated `GaussNoise` (`std_range: [0.02, 0.08]`, sized
for the `[0,1]` scale — see the `GaussNoise` guard in `augmentations.py`), mild
`GaussianBlur`, and a wider affine scale range (`0.75–1.35`, since fiber pixel-width
varies with SEM magnification across the dataset). Elastic transforms are
deliberately omitted — they corrupt thin-mask geometry. **Cost:** negligible.

---

## Tier 3 — structural (biggest levers)

### 8b. Weighted tile sampling with hard-negative mining (`data.tile_sampling`)
An alternative to the static pre-filtered tile set. Instead of dropping most empty
tiles once at startup (`min_foreground_fraction`/`keep_empty_probability`), keep
**all** tiles and resample each epoch: every positive tile plus a fresh draw of
`negative_ratio × #positives` negatives. This gives the coverage of "all tiles"
without the class imbalance, and — unlike a static set — shows the model different
negatives every epoch.

Set `data.tile_sampling: "weighted"` to enable (default `"static"` = previous
behavior, so it's a clean A/B). Knobs:
- `negative_ratio` — negatives per positive per epoch (`1.0` = balanced).
- `hard_negative_fraction` — fraction of the negative draw taken from the **hardest**
  negatives (highest recent training loss) rather than uniformly. `0.0` = pure random
  (a strong, feedback-free baseline); higher biases toward hard negatives the model
  currently gets wrong.
- `hard_negative_warmup_epochs` — random-only epochs before hard mining activates
  (so difficulty estimates have signal first).

How the feedback works: in weighted mode the dataset returns each tile's index,
`training_step` reports that tile's per-sample loss, and `HardNegativeMiningCallback`
pushes the latest per-tile loss into `WeightedTileSampler` each epoch to reweight the
next negative draw. All of this is inert in `"static"` mode. **Cost:** negligible
(one extra detached BCE reduction per step). Recommended once a baseline is stable, to
compare against the static filter.

### 9. The binding constraint is dataset size (28 images, val=4)
Checkpoint selection, early stopping, LR scheduling, and reported metrics all hinge
on **4 validation images** — single-image luck swings `val/tversky` more than most
modeling changes. Two remedies:
- **Active-learning annotation** (`tools/rank_uncertainty.py`): run a trained model
  over an unlabeled candidate pool, rank by uncertainty (mean binary entropy), and
  annotate the most-uncertain images next. Ten well-chosen images beat most
  architecture tweaks combined.
  ```powershell
  python -m fiberseg.tools.rank_uncertainty --config <cfg> --checkpoint <best.ckpt> `
      --images-dir <unlabeled-folder> --out uncertainty_ranking.csv
  ```
- **5-fold cross-validation** for trustworthy model selection/reporting (not yet
  automated — run five configs with different `data.seed` and average, holding a
  test set out). Right now a 0.02 tversky difference between runs is within noise.

### 10. Configurable checkpoint-selection metric (`train.monitor_metric`, `train.monitor_mode`)
Everything used to hardcode `val/tversky` (recall-weighted). If your downstream
target is dice/IoU, select on that: set `monitor_metric: "val/dice"`. The metric
also feeds the LR scheduler and the checkpoint filename (both derived from it).
**Cost:** none.

### 11. Unpin segmentation-models-pytorch (`tools/extract_micronet_weights.py`)
Installing NASA `pretrained-microscopy-models` force-downgrades smp (0.5→0.2.1) and
timm (1.0→0.4.12) environment-wide, because `pmm` pins old versions — yet `pmm`'s
only real contribution is encoder weights it downloads from a URL. To unpin:

```powershell
# with pmm still installed, snapshot the encoder weights locally
python -m fiberseg.tools.extract_micronet_weights --config configs/micronet/unet_resnet50.yaml `
    --out pretrained/micronet_resnet50.pth
# then restore modern smp
pip install "segmentation-models-pytorch>=0.5" "timm>=1.0"
```

You then load the saved encoder weights into a standard smp model (small extension
to `models.create_model` to accept a local weights path). This removes a fragile
dependency and unlocks modern encoders (ConvNeXt, SegFormer/MiT) for sweeps.
**Not executed automatically** — it changes the environment, so it's left as an
explicit opt-in step.

---

## Metric-computation fix (prerequisite for all of the above)

Validation/test metrics are now **micro-averaged over the whole epoch**
(accumulate tp/fp/fn per batch in `validation_step`/`test_step`, compute ratios once
in `on_validation_epoch_end`/`on_test_epoch_end`) instead of averaging per-batch
ratios. On sparse fiber masks the old per-batch average let empty tiles score ~0 and
deflate `val/tversky` (~0.23) far below the true whole-image score (~0.75–0.87).
The new number tracks the real metric and gives checkpoint selection / early
stopping a trustworthy signal. See `lit_module._confusion_counts` /
`_stats_from_counts`.
