# Script reference

All commands assume `cd fiber_unet_trainer` with the `cnn_test` conda env active.
Files fall into three groups: **entry-point scripts** (run directly), **library
modules** (imported only, never run), and **tests**.

## Relational structure

```
fiberseg/__init__.py        <- runs first on every import (env vars, warning filters)
fiberseg/config.py          <- AppConfig / load_config()  (no internal deps)
fiberseg/augmentations.py   <- build_transform()          (no internal deps)
        |
        v
fiberseg/dataset.py         <- FiberDataModule, tiling, disk cache  (config, augmentations)
fiberseg/models.py          <- create_model()                       (config)
        |                            |
        v                            v
fiberseg/lit_module.py      <- FiberSegmentationLitModule  (config, models)
fiberseg/callbacks.py       <- prediction-image loggers    (used by train.py)
        |
        v
fiberseg/train.py  ---------+--> fiberseg/sweep.py (calls train._run_single_training)
    |         |             |
    v         v             v
predict_tiles.py  tools/export_torchscript.py  tools/inspect_checkpoint.py
    |    |          tools/preview_augmentations.py
    |    v          tools/foreground_filter_sweep.py
    | tools/tune_threshold.py     tools/lr_range_test.py
    | tools/rank_uncertainty.py
    |    |
    v    v
predict_all.py (calls tune_threshold.find_best_threshold if --tune-threshold is set,
    default --tune-split val; ALWAYS calls postprocess_masks.postprocess_mask, then
    evaluate_predictions.compute_metrics on both the raw and processed mask; writes
    raw masks + postprocessed masks + one --out-dir/metrics.csv per image (columns
    raw_*/post_*/delta_* per metric) + threshold_info.txt, and threshold_pr_curve.png
    if tuned - no separate tune_threshold/postprocess_masks/evaluate_predictions run
    needed for a predict_all.py output)
    |
    v
tools/add_gt_foreground_fraction.py, tools/median_otsu_coverage.py (both add a column
    to metrics.csv)  -->  tools/merge_sample_overview.py (standalone: pandas only,
                           joins metrics.csv with a sample-metadata sheet)

tools/evaluate_predictions.py and tools/postprocess_masks.py are also standalone
entry points - run either directly against any prediction folder (e.g. from
predict_tiles.py) that wasn't produced by predict_all.py.

tools/label_pairs.py (config only - not a training entry point)
tools/label_fiber_type.py (reuses label_pairs.PairLabeler)
tools/compute_dataset_stats.py, tools/list_split_files.py  (config + dataset only, no checkpoint)
tools/extract_micronet_weights.py                          (config + models only, no dataset)

export_model.py (repo root) --shells out to--> python -m fiberseg.tools.export_torchscript
```

`dataset.py`, `config.py`, `models.py`, `lit_module.py`, `callbacks.py`,
`augmentations.py`, `fiberseg/__init__.py` are **library modules only** — never
run directly, always imported.

## Entry-point scripts

| Script | Run with | Required args | Optional args |
|---|---|---|---|
| `fiberseg/train.py` | `python -m fiberseg.train` | `--config PATH` | `--resume PATH` (checkpoint to resume from), `--profile {simple,advanced,pytorch}` (attach a bottleneck profiler) |
| `fiberseg/sweep.py` | `python -m fiberseg.sweep` | `--config PATH` (must contain a `sweep:` section) | — |
| `fiberseg/predict_tiles.py` | `python -m fiberseg.predict_tiles` | `--config PATH`, `--checkpoint PATH`, `--image PATH`, `--out PATH` | — |
| `fiberseg/predict_all.py` | `python -m fiberseg.predict_all` | `--config PATH`, `--checkpoint PATH`, `--out-dir DIR` | `--suffix STR` (default `_pred.tif`), `--tune-threshold` (re-tune `train.threshold` via `tools.tune_threshold.find_best_threshold`, then use that one threshold for every image), `--tune-split {train,val,test}` (default `val`; never tune on `test` for reporting), `--tune-metric {dice,iou,f2,tversky}` (default `dice`), `--tune-steps N` (default `99`), `--closing-radius`/`--opening-radius`/`--min-object-size`/`--max-hole-size` (post-processing params, same defaults as `postprocess_masks.py`: `1`/`0`/`64`/`64`) — runs tiled prediction on every image/mask pair found via `find_pairs` (same discovery/split rule as training; a matching mask is required), writing each raw mask into a `train`/`validation`/`test` subfolder of `--out-dir`. Every raw prediction is also run through `tools.postprocess_masks.postprocess_mask` automatically, with the processed mask written into a parallel `postprocessed/train`/`validation`/`test` subfolder. Both are scored against ground truth into one `--out-dir/metrics.csv`, whose columns are `raw_*`/`post_*`/`delta_*` (post minus raw) per metric — so neither `evaluate_predictions.py` nor `postprocess_masks.py` need a separate run. Also writes `--out-dir/threshold_info.txt` (tuning on/off, split/metric/steps, threshold used, post-processing params, mean delta) and, when `--tune-threshold` is set, `--out-dir/threshold_pr_curve.png` (precision-recall curve + AUC-PR over the sweep). |
| `fiberseg/tools/export_torchscript.py` | `python -m fiberseg.tools.export_torchscript` | `--config PATH`, `--checkpoint PATH`, `--out-dir DIR` | `--model-name NAME` (default `fiber_unet`), `--device {cpu,cuda}` (default `cpu`), `--no-verify` (skip output-diff check) |
| `fiberseg/tools/inspect_checkpoint.py` | `python -m fiberseg.tools.inspect_checkpoint` | `--config PATH`, `--checkpoint PATH` | `--out-dir DIR` (default `inspection_outputs`), `--split {train,val,test}` (default `test`), `--max-images N` (default `8`), `--threshold FLOAT` (default: config's `train.threshold`) |
| `fiberseg/tools/preview_augmentations.py` | `python -m fiberseg.tools.preview_augmentations` | `--config PATH` | `--out DIR` (default `augmentation_preview`), `--n-images N` (default `3`), `--n-aug N` (default `5`), `--crop-size N` (default `1024`), `--dpi N` (default `300`), `--save-raw-crops` |
| `fiberseg/tools/foreground_filter_sweep.py` | `python -m fiberseg.tools.foreground_filter_sweep` | `--config PATH` | `--patch-sizes N [N ...]`, `--strides N [N ...]`, `--foreground-fractions F [F ...]`, `--keep-empty-probability F`, `--split {train,val,test,all}`, `--boundary-margin F` (default `0.02`), `--n-examples N` (default `8`), `--out DIR` (default `foreground_filter_sweep`), `--dpi N` |
| `fiberseg/tools/lr_range_test.py` | `python -m fiberseg.tools.lr_range_test` | `--config PATH` | `--min-lr F` (default `1e-8`), `--max-lr F` (default `1.0`), `--num-training N` (default `100`), `--mode {exponential,linear}` (default `exponential`), `--early-stop-threshold F` (default `4.0`; `0` disables), `--out PATH` (default `lr_range_test.png`) |
| `fiberseg/tools/tune_threshold.py` | `python -m fiberseg.tools.tune_threshold` | `--config PATH`, `--checkpoint PATH` | `--split {val,test,train}` (default `val`; never tune on `test` for reporting), `--metric {dice,iou,f2,tversky}` (default `dice`), `--steps N` (default `99`), `--plot PATH` (optional: save a precision-recall curve with AUC-PR over the sweep) — sweeps `train.threshold` on full-resolution tiled-inference probability maps and reports the value maximizing the chosen metric. No retrain needed. Also exposes `sweep_thresholds()`/`find_best_threshold()`/`plot_pr_curve()` as reusable functions (used by `predict_all.py --tune-threshold`), not just a CLI. |
| `fiberseg/tools/rank_uncertainty.py` | `python -m fiberseg.tools.rank_uncertainty` | `--config PATH`, `--checkpoint PATH`, `--images-dir PATH` (folder of UNLABELED candidate images) | `--glob PATTERN` (default `*.tif`), `--out PATH` (default `uncertainty_ranking.csv`) — scores each image by prediction uncertainty (mean entropy, boundary-band fraction) and writes a CSV ranked most-uncertain first, for active-learning annotation prioritization. |
| `fiberseg/tools/compute_dataset_stats.py` | `python -m fiberseg.tools.compute_dataset_stats` | `--config PATH` | `--write` (patch the computed `norm_mean`/`norm_std` + `image_normalization: "dataset"` directly into the config's `data:` block in place; default is print-only) — computes per-channel mean/std over the training split only, for `image_normalization: "dataset"`. |
| `fiberseg/tools/evaluate_predictions.py` | `python -m fiberseg.tools.evaluate_predictions` | `--config PATH`, `--pred-dir DIR` (folder of prediction masks, e.g. from `predict_all.py`) | `--suffix STR` (default `_pred.tif`, must match the suffix predictions were saved with), `--out PATH` (default `<pred-dir>/metrics.csv`) — per-image accuracy/precision/recall/specificity/dice/iou/tversky/f2 vs. ground truth, written to CSV. Looks under `--pred-dir/<train\|validation\|test>/` first (matching `predict_all.py`'s split subfolders), falling back to flat `--pred-dir`. |
| `fiberseg/tools/extract_micronet_weights.py` | `python -m fiberseg.tools.extract_micronet_weights` | `--config PATH` (must have `model.encoder_weights: micronet`) | `--out PATH` (default `pretrained/micronet_<encoder>.pth`) — snapshots just the MicroNet encoder weights locally so the `pretrained-microscopy-models` package (and the smp/timm downgrade it forces) is no longer needed at train time. |
| `fiberseg/tools/postprocess_masks.py` | `python -m fiberseg.tools.postprocess_masks` | `--config PATH`, `--pred-dir DIR` (e.g. `predict_all.py`'s `--out-dir`) | `--out-dir DIR` (default `<pred-dir>/postprocessed`), `--suffix STR` (default `_pred.tif`), `--closing-radius N` (default `1`; bridges small gaps in broken fibers, 0 disables), `--opening-radius N` (default `0`; smooths jagged edges, can erode thin fibers if too large), `--min-object-size N` (default `64`; removes small noise blobs, 0 disables), `--max-hole-size N` (default `64`; fills small interior holes, 0 disables) — classical (non-learned) morphological cleanup via `postprocess_mask()` (closing → opening → remove-small-objects → fill-small-holes), applied to every prediction found in `--pred-dir` (same split-subfolder/flat lookup as `evaluate_predictions.py`), writing processed masks into the same `train`/`validation`/`test` layout under `--out-dir` and re-scoring them into `--out-dir/metrics.csv`; prints a before/after mean-metric comparison if `--pred-dir/metrics.csv` already exists. |
| `fiberseg/tools/fiber_gap_repair.py` | `python -m fiberseg.tools.fiber_gap_repair` | `--config PATH`, `--pred-dir DIR` (e.g. `predict_all.py`'s `--out-dir`, raw or `postprocessed/`) | `--out-dir DIR` (default `<pred-dir>/gap_repaired`), `--suffix STR` (default `_pred.tif`), `--max-gap N` (default `15`; max px a link may bridge), `--max-angle-deg F` (default `30`; the collinearity gate — both fragments' local tangents must point toward each other within this angle, on both sides, or the link is rejected), `--min-object-size N` (default `4`), `--connector-width N` (default `1`), `--no-oriented-closing`, `--closing-radius N` (default `4`), `--closing-bins N` (default `8`) — repairs fragmented ("broken") thin fibre masks so length/aspect-ratio-based WHO/ISO counting isn't measuring pieces of what should be one whole fibre: an optional oriented-closing pre-pass (`oriented_closing()`, direction-aware, unlike isotropic closing) bridges the shortest gaps, then every fragment's skeleton endpoints are found (`skeleton_endpoints()`) and candidate links are gated by gap distance AND the collinearity check above (this is what stops two distinct nearby fibres from being wrongly merged — verified by tests), connected via a minimum-cost path (`route_through_array`, straight line if no probability map given) and accepted globally cheapest-first with each endpoint usable once. Also exposes `hysteresis_threshold_mask()` and `enhance_ridges()` (Frangi/Sato) as separate probability-map-based *thresholding-time* alternatives (build a less-fragmented mask in the first place, rather than repairing one after the fact) — these need the continuous prediction probability, not the saved binary mask, so aren't wired into this CLI. Writes raw/repaired/delta metrics into one combined `--out-dir/metrics.csv`, same convention as `predict_all.py`. On real predictions this consistently raised recall (fragments reconnected) at a small Dice/precision cost (bridged pixels aren't always pixel-exact) — tune `--max-gap`/`--max-angle-deg` to control that trade-off. |
| `fiberseg/tools/add_gt_foreground_fraction.py` | `python -m fiberseg.tools.add_gt_foreground_fraction` | `--config PATH`, `--csv PATH` (metrics CSV with an `image` column) | `--out PATH` (default: overwrite `--csv` in place) — adds a `GT Foreground Fraction (%)` column computed from each image's ground-truth mask. |
| `fiberseg/tools/median_otsu_coverage.py` | `python -m fiberseg.tools.median_otsu_coverage` | `--config PATH`, `--csv PATH` (metrics CSV with an `image` column), `--mask-out-dir DIR` | `--median-radius N` (default `4`), `--num-passes N` (default `4`), `--hole-dilation-radius N` (default `50`) — classical median-filter + 3-class-Otsu particle-coverage estimate per image (hole-aware, for SEM filter substrate images); saves QC masks and adds a `Median Otsu Coverage (%)` column, `--out PATH` (default: overwrite `--csv` in place) |
| `fiberseg/tools/fiber_contrast_metrics.py` | `python -m fiberseg.tools.fiber_contrast_metrics` | `--masks-dir DIR`, `--images-dir DIR` (matching source images; no `--config` needed) | `--mask-pattern STR` (default `{stem}_mask.tif`), `--image-glob STR` (default `*.tif`), `--dilation-size N` (default `5`), `--gap-size N` (default `2`), `--min-object-size N` (default `1`, no filtering), `--level {object,image}` (default `object`), `--out PATH` (default `fiber_contrast_metrics.csv`) — object-level (per-connected-component) SNR/CNR/Weber contrast from each ground-truth mask against its source image: `mu_F`/`sigma_B` computed over the fibre pixels and a local background ring (mask dilated by `dilation_size` px, offset out by `gap_size` px first to avoid fibre-edge bleed, and excluding every fibre pixel anywhere in the image so a neighbouring fibre inside the ring can't contaminate it). `sigma_B` is the MAD-SD (`1.4826 * median(\|x - median(x)\|)`), not plain std, so a stray bright pixel in the ring doesn't inflate it. `--level image` instead returns one row per image, with `*_mean`/`*_median` (aggregated per-object stats) and `*_pooled` (every fibre pixel in the image pooled into one region with one ring around the whole mask, computed by `compute_pooled_image_metrics()`) columns side by side, since "per-image SNR" is ambiguous between those two readings. Also exposes `compute_object_metrics()` (one image/mask pair -> list of per-object dicts), `compute_pooled_image_metrics()` (one image/mask pair -> single whole-mask dict), and `compute_fiber_contrast_metrics()` (a whole folder -> DataFrame) as reusable functions, e.g. for a notebook. Note: densely-packed/touching fibres merge into one connected component under 8-connectivity, so object-level "per fibre" only cleanly separates in images where fibres don't touch in the mask; per-object dilation/ring computation is done on a small crop around each object's bounding box, not the full image, so it stays fast even on large (thousands-of-px) SEM images with many fibres. |
| `fiberseg/tools/merge_sample_overview.py` | `python -m fiberseg.tools.merge_sample_overview` | `--metrics-csv PATH` (e.g. from `evaluate_predictions.py`), `--overview-xlsx PATH` | `--sheet NAME_OR_INDEX` (default first sheet), `--sample-col STR` (default `Sample`), `--location-col STR` (default `Location`), `--image-col STR` (default `image`), `--out PATH` (default `<metrics-csv-stem>_with_sample_info.csv`) — joins per-image metrics with sample metadata by matching filename Sample/Site prefixes; unresolvable rows are left blank and flagged via `match_status` rather than guessed. Standalone (pandas only, no config/checkpoint needed). |
| `fiberseg/tools/list_split_files.py` | `python -m fiberseg.tools.list_split_files` | `--config PATH` | `--out PATH` (write result as JSON; default: print to stdout) — reports which image filenames were assigned to each train/val/test split. |
| `fiberseg/tools/label_pairs.py` | `python -m fiberseg.tools.label_pairs` | `--config PATH` | `--images-dir PATH`, `--masks-dir PATH`, `--image-glob PATTERN`, `--mask-pattern PATTERN` (override the config's `data:` section), `--labels-csv PATH` (default `notebooks/pair_labels.csv`), `--relabel-all`. Opens a native (TkAgg) zoomable window; keyboard `g`/`b`/`r`/`u`/`q` label each pair good/bad/redo, undo, or quit. |
| `fiberseg/tools/label_fiber_type.py` | `python -m fiberseg.tools.label_fiber_type` | `--config PATH` | Same override/`--relabel-all` args as `label_pairs.py` (reuses its `PairLabeler`), `--labels-csv PATH` (default `notebooks/fiber_type_labels.csv`). Same zoomable-window interaction, but keys `c`/`a`/`u`/`r`/`q` label Chrysotile/Amphibole/undo/reset-zoom/quit instead. |
| `export_model.py` (repo root) | `python export_model.py` | none — fully interactive (prompts for checkpoint/config/output/device), then shells out to `export_torchscript.py` | — |

## Tests (`tests/`)

Two kinds live side by side — check for a `def test_*` function to tell them apart.

**Pytest suite** — run with `pytest`, or `pytest tests/<file>.py::<test_name>` for one test. No CLI args.
`test_augmentations.py`, `test_config_validation.py`, `test_dataset_foreground_filter.py`,
`test_dataset_normalization.py`, `test_gpu_setup.py`, `test_sweep_expansion.py`.

**Standalone diagnostic scripts** — run directly with `python tests/<file>.py`, *not* collected by pytest (no `test_*` functions):

| Script | Run with | Args |
|---|---|---|
| `test_micronet_forward.py` | `python tests/test_micronet_forward.py` | none (hardcoded to `configs/cnn_micronet_resnet50.yaml`) |
| `check_model_logits.py` | `python tests/check_model_logits.py` | none (hardcoded to `configs/cnn_micronet_resnet50.yaml`) — asserts `create_model()` returns raw logits, not probabilities |
| `check_training_speed.py` | `python tests/check_training_speed.py` | `--config PATH` (required), `--num-batches N` (default `50`) — benchmarks dataloader and train-step throughput |
| `debug_probability_response.py` | `python tests/debug_probability_response.py` | `--config PATH` (required), `--checkpoint PATH` (optional), `--split {train,val,test}` (default `val`), `--out PATH` (default `debug_probability_response.png`) |
