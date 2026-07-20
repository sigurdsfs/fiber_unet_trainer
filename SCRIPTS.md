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
                   tools/preview_augmentations.py
                   tools/foreground_filter_sweep.py
                   tools/lr_range_test.py
                   tools/label_pairs.py (config only - not a training entry point)

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
| `fiberseg/tools/export_torchscript.py` | `python -m fiberseg.tools.export_torchscript` | `--config PATH`, `--checkpoint PATH`, `--out-dir DIR` | `--model-name NAME` (default `fiber_unet`), `--device {cpu,cuda}` (default `cpu`), `--no-verify` (skip output-diff check) |
| `fiberseg/tools/inspect_checkpoint.py` | `python -m fiberseg.tools.inspect_checkpoint` | `--config PATH`, `--checkpoint PATH` | `--out-dir DIR` (default `inspection_outputs`), `--split {train,val,test}` (default `test`), `--max-images N` (default `8`), `--threshold FLOAT` (default: config's `train.threshold`) |
| `fiberseg/tools/preview_augmentations.py` | `python -m fiberseg.tools.preview_augmentations` | `--config PATH` | `--out DIR` (default `augmentation_preview`), `--n-images N` (default `3`), `--n-aug N` (default `5`), `--crop-size N` (default `1024`), `--dpi N` (default `300`), `--save-raw-crops` |
| `fiberseg/tools/foreground_filter_sweep.py` | `python -m fiberseg.tools.foreground_filter_sweep` | `--config PATH` | `--patch-sizes N [N ...]`, `--strides N [N ...]`, `--foreground-fractions F [F ...]`, `--keep-empty-probability F`, `--split {train,val,test,all}`, `--boundary-margin F` (default `0.02`), `--n-examples N` (default `8`), `--out DIR` (default `foreground_filter_sweep`), `--dpi N` |
| `fiberseg/tools/lr_range_test.py` | `python -m fiberseg.tools.lr_range_test` | `--config PATH` | `--min-lr F` (default `1e-8`), `--max-lr F` (default `1.0`), `--num-training N` (default `100`), `--mode {exponential,linear}` (default `exponential`), `--early-stop-threshold F` (default `4.0`; `0` disables), `--out PATH` (default `lr_range_test.png`) |
| `fiberseg/tools/label_pairs.py` | `python -m fiberseg.tools.label_pairs` | `--config PATH` | `--images-dir PATH`, `--masks-dir PATH`, `--image-glob PATTERN`, `--mask-pattern PATTERN` (override the config's `data:` section), `--labels-csv PATH` (default `notebooks/pair_labels.csv`), `--relabel-all`. Opens a native (TkAgg) zoomable window; keyboard `g`/`b`/`r`/`u`/`q` label each pair good/bad/redo, undo, or quit. |
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
