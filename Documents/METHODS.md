# Methods: Deep-Learning Segmentation of Fibers in SEM Micrographs

## Summary

We trained a convolutional encoder–decoder network to perform binary semantic
segmentation of fiber structures in grayscale scanning electron microscopy (SEM)
images of sampled air-filter deposits. The model is intended to assist — not
silently replace — the manual fiber-counting workflow used in occupational
asbestos-exposure assessment (filter loading, WHO-criteria fiber counts, and
airborne concentration / time-weighted-average exposure reporting). Training
and evaluation are fully config-driven and tracked in MLflow, so every number
reported below traces back to an exact configuration and code state.

## 1. Dataset and splits

The training corpus consists of 298 image/mask pairs of SEM micrographs
(one image excluded for a missing corresponding mask). Each image is a
single-channel grayscale micrograph; each mask is a binary image where a
pixel value greater than zero denotes fiber and zero denotes background.
Mask files are matched to images by filename pattern (`{stem}_mask.tif`).

Pairs are split **by source image**, not by tile, into 70% train / 15%
validation / 15% test (209 / 45 / 44 images) using a seeded random shuffle
(seed 42). Splitting by image before tiling is deliberate: it prevents tiles
from the same source micrograph from leaking across train/validation/test,
which would otherwise inflate validation/test performance.

## 2. Preprocessing and tiling

**Contrast normalization.** Each source image is normalized once, at full
resolution, by clipping to its 1st and 99.5th percentile intensity and
rescaling to `[0, 1]`. Percentile (rather than min–max) normalization is used
because SEM images commonly contain a small number of very bright outlier
pixels (charging artifacts, detector saturation) that would otherwise dominate
a naive min–max stretch. This normalized array is cached to disk (memory-mapped
`.npy`) so every tile drawn from a given image shares an identical contrast
stretch, and the same normalization is applied verbatim at inference time.

**Tiling.** Each normalized image is decomposed into overlapping square tiles
via a sliding window (default 512×512 px, 384 px stride — 25% overlap). The
final row/column of tiles is snapped to the image edge so the entire image is
always covered, even when its dimensions are not an exact multiple of the
stride.

**Sparse-mask handling.** Fiber pixels are a small minority of total image
area, so a large majority of naively generated tiles are entirely background.
During training (train split only), tiles below a minimum foreground fraction
(default 0.1%) are randomly kept with a fixed low probability (default 20%)
rather than discarded outright — this reduces the volume of uninformative
background tiles without eliminating the model's exposure to hard negatives.
An alternative dynamic mode exists (per-epoch resampling weighted toward the
model's current hardest negatives, i.e. background tiles it currently
misclassifies) but is not used in the production recipe.

**Channel handling and standardization.** The single grayscale channel is
replicated to 3 channels so it is compatible with ImageNet/MicroNet-pretrained
encoders. After augmentation, tiles are standardized with the ImageNet
per-channel mean/std (matching the distribution the pretrained encoder was
fine-tuned on); this standardization is applied identically in the training
dataset and in the tiled-inference code path.

## 3. Data augmentation

Applied to the training split only (validation/test are unaugmented), via
Albumentations:

| Transform | Parameters | Probability |
|---|---|---|
| Horizontal flip | — | 0.5 |
| Vertical flip | — | 0.5 |
| 90° rotation | — | 0.5 |
| Affine | scale 0.75–1.35, translate ±5%, rotate ±15° | 0.5 |
| Brightness/contrast jitter | ±0.15 | 0.4 |
| Gaussian noise | σ 0.02–0.08 (on the [0,1] scale) | 0.3 |
| Gaussian blur | kernel 3–5 | 0.2 |

The wide affine scale range specifically accounts for fiber pixel-width
varying with SEM magnification across the source dataset. Output is re-clipped
to `[0, 1]` after augmentation, since brightness/noise transforms can push
values slightly outside the normalized range.

## 4. Model architecture

The segmentation backbone is a U-Net (`segmentation_models_pytorch`) with a
ResNet-50 encoder. Encoder weights are initialized from NASA's **MicroNet**
pretrained-microscopy-models corpus — a domain-specific pretraining corpus of
microscopy imagery, rather than natural-image ImageNet — under the hypothesis
that microscopy-domain pretraining transfers better to SEM textures than
natural-image pretraining. The training/inference code is architecture-agnostic:
any `segmentation_models_pytorch` decoder family (U-Net, U-Net++, FPN,
DeepLabV3+) and encoder is swappable via configuration without code changes,
and standard ImageNet-pretrained or from-scratch encoders are supported as
alternatives to MicroNet.

The model outputs a single-channel raw logit map. Any built-in output
activation (sigmoid/softmax) present in third-party model definitions is
explicitly stripped, so sigmoid is applied only inside the loss, metric, and
inference code — this "raw logits" contract is enforced uniformly regardless
of which backend produced the model.

## 5. Loss function

The loss is a weighted sum of three terms, computed on sigmoid probabilities
of the raw logits:

**(a) Binary cross-entropy** (pixelwise, standard).

**(b) Focal Tversky loss.** The soft Tversky index is

```
TI = (TP + ε) / (TP + α·FP + β·FN + ε)
```

with TP/FP/FN computed as probability-weighted sums (not binarized), α = 0.3,
β = 0.7, ε = 1e-7. Weighting β > α penalizes missed fiber pixels (false
negatives) roughly 2.3× more heavily than false positives — appropriate given
that under-segmenting a thin fiber is a more consequential error than mild
over-segmentation for downstream fiber counting. The focal form,
`(1 − TI)^γ` with γ = 0.75, additionally up-weights harder (lower-TI)
examples during training.

**(c) Soft centerline-Dice (clDice) topology term**, weighted 0.3 in the total
loss. A differentiable morphological skeleton is computed for both the
predicted probability map and the ground-truth mask via iterative erosion
(implemented as negated max-pooling, 5 iterations) and re-dilation; clDice is
the harmonic mean of (i) the fraction of the predicted skeleton that lies
inside the true mask and (ii) the fraction of the true skeleton that lies
inside the predicted mask. This term directly rewards connected, unbroken
fiber topology, addressing a failure mode of pure overlap losses (BCE, Dice,
Tversky), which tolerate a single fiber being predicted as several
disconnected fragments as long as total pixel overlap is high.

Total training loss: `L = BCE + (1 − TI)^γ + 0.3 · (1 − clDice)`.

## 6. Training procedure

- **Optimizer:** AdamW, weight decay 1e-4.
- **Differential learning rate:** base learning rate 2×10⁻⁴ applied to the
  randomly-initialized decoder/head; the pretrained encoder receives this
  rate scaled by 0.1 (2×10⁻⁵), so the transferred features are fine-tuned
  gently while the fresh decoder learns quickly.
- **LR schedule:** `ReduceLROnPlateau` (factor 0.5, patience 4 epochs) driven
  by the monitor metric (Section 7); cosine annealing is supported as an
  alternative.
- **Early stopping:** patience 20 epochs, minimum delta 1×10⁻³, on the same
  monitor metric.
- **Budget:** up to 400 epochs (early stopping typically halts training well
  before this ceiling).
- **Batch size:** 8 tiles of 512×512 px.
- Stochastic Weight Averaging is supported but disabled in the production
  recipe.

## 7. Training-control metric

Checkpoint selection, early stopping, and LR scheduling are driven by a
**threshold-free** validation metric, `val/soft_tversky` — the same Tversky
formula as Section 5(b), but accumulated as probability-weighted TP/FP/FN
sums over the *entire* validation epoch (not averaged per-batch, which would
let empty tiles trivially score near zero and deflate the true whole-image
score on such sparse masks) and never binarized at a threshold. This is a
deliberate design choice: it decouples training-time model selection from
`train.threshold`, a deployment decision that is not yet calibrated during
training (see Section 8). Hard, thresholded analogues of every metric
(`val/dice`, `val/tversky`, etc., at the eventual deployment threshold) are
logged in parallel for human-readable, deployment-realistic reporting, but do
not drive any training decision.

## 8. Post-hoc threshold calibration

After training, the sigmoid-probability decision threshold is calibrated —
rather than left at an arbitrary default of 0.5 — by sweeping it across ~99
values evaluated **only on the validation split** (never test, to avoid
leakage into the held-out evaluation set) and selecting the value that
maximizes a chosen metric (Dice by default). A precision–recall curve and its
AUC are reported for this sweep; precision–recall (rather than ROC) is used
because fiber masks are strongly background-dominated, and ROC-AUC would look
misleadingly high under that class imbalance — the same reasoning that
motivates using Dice/IoU/Tversky over raw pixel accuracy throughout this
pipeline. The single resulting threshold is then applied uniformly across
every split at inference time.

## 9. Inference

Full-resolution images are tiled using the same patch size/stride as training,
so the effective receptive field and context seen by the model matches
training exactly. Overlapping tile probabilities are blended with a 2D
Gaussian window (following the nnU-Net sliding-window scheme), which
down-weights each tile's less-reliable border pixels relative to its center
and smooths seams between adjacent tiles. Edge tiles are reflect-padded
(rather than zero-padded) so they are not artificially biased toward
"background" by a black border.

Optional 8× dihedral test-time augmentation (TTA) — every combination of
flip/90° rotation, each inverse-transformed and probability-averaged — is
available as a no-retrain accuracy gain (empirically on the order of +1–2
Dice) at 8× inference cost; it is left off during iterative development and
enabled only for final reported predictions. The final binary mask is the
blended probability map thresholded at the calibrated value from Section 8.

## 10. Post-processing

Every predicted mask additionally passes through a fixed, classical (non-
learned) morphological cleanup, applied in this order:

1. Binary closing (disk radius 1) — run *first* so it can bridge small,
   genuine gaps in a single fiber before the next step would otherwise
   mistake the resulting fragments for separate noise specks.
2. Binary opening (disabled by default, radius 0) — available to smooth
   jagged edges, but disabled because it risks eroding away genuinely thin
   fibers.
3. Removal of connected components ≤64 px.
4. Filling of interior holes ≤64 px.

Both the raw and post-processed masks are scored independently against
ground truth, and the per-image delta (post − raw) for every metric is
reported alongside the absolute values, so the contribution of this
deterministic cleanup stage is always visible rather than assumed.

## 11. Evaluation metrics

For every image, the full confusion matrix (TP/FP/FN/TN, pixelwise) is
computed against ground truth, from which accuracy, precision, recall,
specificity, Dice, IoU, Tversky (α = 0.3, β = 0.7 — the same asymmetric
weighting used in the loss, so optimization and reporting stay consistent),
and F2 (a recall-weighted F-beta score) are derived. Metrics are computed
identically for raw and post-processed predictions, across every split, and
written to a single CSV for auditability.

## 12. Experiment tracking and reproducibility

Every training run logs its fully resolved configuration, per-step and final
metrics, and periodic/best qualitative prediction images to a local MLflow
tracking server, keyed by experiment name. The entire pipeline — data split,
preprocessing, model, loss, training schedule, inference, and
post-processing — is defined by one declarative YAML configuration per
experiment (Table 1), so reproducing a reported result requires no code
changes, only re-running training against the same config.

## 13. Application context

The source SEM images originate from air-filter samples collected for
occupational asbestos-exposure monitoring. Segmented fiber masks feed
downstream quantification — filter loading, WHO-criteria fiber counting, and
airborne fiber concentration / time-weighted-average exposure reporting —
that otherwise relies entirely on manual microscopist counting under
magnification. The model's intended role is to **assist and accelerate**,
not silently replace, that expert workflow: the held-out test split has no
influence on threshold calibration (Section 8), and companion tooling in the
same repository supports manual per-image quality review (mask quality,
filter loading level, and asbestos presence) so segmentation output remains
auditable against expert judgment before being used in any exposure-limit-
relevant reporting.

## Table 1. Production configuration

| Setting | Value |
|---|---|
| Architecture / encoder | U-Net, ResNet-50, MicroNet-pretrained |
| Input channels | 3 (grayscale replicated) |
| Patch size / stride | 512 / 384 px (25% overlap) |
| Batch size | 8 |
| Split (train/val/test), seed | 70/15/15, seed 42 |
| Loss | BCE + Focal Tversky (α=0.3, β=0.7, γ=0.75) + 0.3 × clDice (5 iters) |
| Optimizer | AdamW, lr 2×10⁻⁴ (decoder) / 2×10⁻⁵ (encoder), wd 1×10⁻⁴ |
| LR schedule | ReduceLROnPlateau, factor 0.5, patience 4 |
| Early stopping | patience 20, min Δ 1×10⁻³ |
| Max epochs | 400 |
| Monitor metric | `val/soft_tversky` (threshold-free, maximize) |
| Inference blending | Gaussian window, reflect padding |
| Test-time augmentation | 8× dihedral (final predictions only) |
| Post-processing | close (r=1) → open (r=0, off) → remove objects ≤64px → fill holes ≤64px |
