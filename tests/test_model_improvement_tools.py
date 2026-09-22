"""Tests for the evaluation / pixel-size / run-comparison tooling."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from fiberseg.config import load_config
from fiberseg.tools.compare_runs import compare, paired_bootstrap
from fiberseg.tools.evaluate_predictions import FIELDNAMES, compute_metrics, tolerance_metrics
from fiberseg.tools.pixel_size import pixel_size_nm
from fiberseg.tools.standardize_pixel_size import (
    resample_image,
    resample_mask,
    standardize,
)
from fiberseg.tools.tune_threshold import (
    _best_index,
    _threshold_grid,
    _tuning_grid,
    average_precision,
)
from fiberseg.train import _expand_sweep_configs, _training_seed

REPO = Path(__file__).resolve().parents[1]


def _line_mask(shape=(64, 64), row=32, width=3):
    m = np.zeros(shape, bool)
    m[row - width // 2: row - width // 2 + width, 8:56] = True
    return m


# ---- tolerance metrics -----------------------------------------------------------

def test_tolerance_metrics_forgive_small_offsets_but_dice_does_not():
    gt = _line_mask(row=32)
    pred = _line_mask(row=34)  # same fiber, drawn 2 px lower
    strict = compute_metrics(pred, gt)
    assert strict["dice"] < 0.5
    assert strict["tol_f1"] == pytest.approx(1.0, abs=1e-6)
    # clDice is NOT offset-tolerant: a 3-px fiber shifted 2 px puts its skeleton outside GT.
    assert strict["cldice"] < 0.5


def test_cldice_forgives_width_disagreement():
    gt = _line_mask(width=1)
    pred = _line_mask(width=5)  # same fiber, annotated much wider
    m = compute_metrics(pred, gt)
    assert m["dice"] < 0.4
    assert m["cldice"] > 0.95  # not exactly 1: skeletons trim slightly at fiber ends


def test_tolerance_metrics_still_penalize_a_phantom_fiber():
    gt = _line_mask(row=16)
    pred = gt | _line_mask(row=48)  # correct fiber + a far-away false one
    m = tolerance_metrics(pred, gt)
    assert m["tol_recall"] == pytest.approx(1.0, abs=1e-6)
    assert m["tol_precision"] == pytest.approx(0.5, abs=0.02)


def test_tolerance_metrics_empty_masks_do_not_crash():
    empty = np.zeros((16, 16), bool)
    m = tolerance_metrics(empty, empty)
    assert all(np.isfinite(v) for v in m.values())


def test_fieldnames_cover_compute_metrics_output():
    m = compute_metrics(_line_mask(), _line_mask())
    assert list(m) == FIELDNAMES[2:]


# ---- PR curve ----------------------------------------------------------------------

def test_threshold_grid_reaches_the_extremes_and_keeps_tuning_grid():
    grid = _threshold_grid(99)
    assert grid.min() < 1e-6 and grid.max() > 1 - 1e-6
    assert np.isin(_tuning_grid(99), grid).all()


def test_best_index_ignores_tail_thresholds():
    grid = _threshold_grid(9)
    score = np.zeros_like(grid)
    score[-1] = 10.0  # best score sits on a tail threshold that tuning must not pick
    tuned = _tuning_grid(9)
    score[np.flatnonzero(np.isin(grid, tuned))[3]] = 1.0
    assert grid[_best_index(grid, score, 9)] == pytest.approx(tuned[3])


def test_average_precision_perfect_and_partial_curves():
    recall = np.array([1.0, 0.5])      # thresholds low -> high
    assert average_precision(np.array([1.0, 1.0]), recall) == pytest.approx(1.0)
    # Half the recall range at precision 1, half at 0.5.
    assert average_precision(np.array([0.5, 1.0]), recall) == pytest.approx(0.75)


# ---- pixel-size standardization -------------------------------------------------

def test_resample_image_preserves_dtype_and_scales_shape():
    img = (np.arange(100 * 80) % 65535).astype(np.uint16).reshape(100, 80)
    out = resample_image(img, 0.5)
    assert out.dtype == np.uint16 and out.shape == (50, 40)
    assert resample_image(img, 2.0).shape == (200, 160)


def test_resample_mask_keeps_one_pixel_fibers_when_shrinking():
    mask = np.zeros((120, 120), np.uint8)
    mask[61, :] = 255                    # 1-px fiber
    out = resample_mask(mask, 1 / 3.5)   # e.g. 14.3 -> 50 nm/px
    assert out.dtype == np.uint8 and set(np.unique(out)) <= {0, 255}
    assert (out > 0).any(axis=0).all()  # the fiber survives across its full length


def test_standardize_writes_new_folder_and_keeps_split(tmp_path):
    src_i, src_m = tmp_path / "Images", tmp_path / "Masks"
    src_i.mkdir()
    src_m.mkdir()
    rng = np.random.default_rng(0)
    names = ["A1_20.0nm_x.tif", "A2_50.0nm_x.tif", "B3 - Site 1.tif"]  # resample / at target / unknown
    for n in names:
        tifffile.imwrite(src_i / n, rng.integers(0, 255, (60, 80), dtype=np.uint8))
        m = np.zeros((60, 80), np.uint8)
        m[30, :] = 255
        tifffile.imwrite(src_m / n.replace(".tif", "_mask.tif"), m)
    before = {p.name: p.stat().st_mtime for p in list(src_i.iterdir()) + list(src_m.iterdir())}

    out = tmp_path / "PixelSize_50nm"
    df = standardize(src_i, src_m, out, 50.0).set_index("image")

    assert {p.name: p.stat().st_mtime for p in list(src_i.iterdir()) + list(src_m.iterdir())} == before
    assert sorted(p.name for p in (out / "Images").iterdir()) == sorted(names)
    assert df.loc["A1_20.0nm_x.tif", "status"] == "resampled"
    assert tifffile.imread(out / "Images" / "A1_20.0nm_x.tif").shape == (24, 32)
    assert pixel_size_nm(str(out / "Images" / "A1_20.0nm_x.tif")).nm == pytest.approx(50.0)
    assert df.loc["A2_50.0nm_x.tif", "status"] == "copied_at_target"
    assert df.loc["B3 - Site 1.tif", "status"] == "unresolved_copied"
    assert (out / "unresolved_pixel_sizes.xlsx").exists()

    # Overrides fill in the unknown one on a re-run.
    ov = tmp_path / "ov.csv"
    pd.DataFrame({"image": ["B3 - Site 1.tif"], "pixel_size_nm": [25.0]}).to_csv(ov, index=False)
    from fiberseg.tools.standardize_pixel_size import load_overrides
    df2 = standardize(src_i, src_m, out, 50.0, overrides=load_overrides(ov), overwrite=True)
    row = df2.set_index("image").loc["B3 - Site 1.tif"]
    assert row["source"] == "override" and row["status"] == "resampled"
    assert not (out / "unresolved_pixel_sizes.xlsx").exists()  # stale sheet removed


# ---- run comparison ------------------------------------------------------------

def test_paired_bootstrap_detects_consistent_gain_and_not_noise():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.4, 0.9, 45)
    diff, lo, hi, _ = paired_bootstrap(a, a + 0.03)
    assert diff == pytest.approx(0.03) and lo > 0
    _, lo, hi, _ = paired_bootstrap(a, a + rng.normal(0, 0.05, 45))
    assert lo < 0 < hi


def test_compare_reports_pooled_dice_when_counts_present():
    a = pd.DataFrame({"dice": [0.5, 0.6], "tp": [10, 20], "fp": [5, 5], "fn": [5, 5]})
    b = pd.DataFrame({"dice": [0.6, 0.7], "tp": [12, 22], "fp": [4, 4], "fn": [3, 3]})
    table = compare(a, b, ["dice"], n_boot=200).set_index("metric")
    assert "pooled dice" in table.index
    assert table.loc["mean dice", "B-A"] == pytest.approx(0.1)


# ---- train.seed ----------------------------------------------------------------

def test_train_seed_sweep_keeps_split_seed():
    cfg = load_config(REPO / "configs" / "proxy" / "resnet34_baseline.yaml")
    runs = _expand_sweep_configs(cfg)
    assert [_training_seed(r) for r in runs] == [1, 2, 3]
    assert {r.data.seed for r in runs} == {42}
    cfg.train.seed = None
    assert _training_seed(cfg) == cfg.data.seed


@pytest.mark.parametrize("name", [
    "proxy/resnet34_baseline.yaml",
    "proxy/resnet34_balanced_loss.yaml",
    "proxy/resnet34_pixelstd_50nm.yaml",
    "micronet/unetPlus_balanced_loss.yaml",
    "micronet/unetPlus_pixelstd_50nm.yaml",
])
def test_new_configs_load(name):
    cfg = load_config(REPO / "configs" / name)
    assert cfg.train.precision == "bf16-mixed"
    assert cfg.train.monitor_metric == "val/soft_dice"
    assert cfg.inference.tta is True


def test_pixel_size_reads_imagej_info_property(tmp_path):
    p = tmp_path / "imagej_resaved.tif"
    info = ("ImageDescription: <Image><PixelWidth_um>0.0508854209703884</PixelWidth_um>"
            "</Image>\n")
    tifffile.imwrite(p, np.zeros((8, 8), np.uint8), imagej=True,
                     metadata={"unit": "inch", "Info": info})
    ps = pixel_size_nm(str(p))
    assert ps.nm == pytest.approx(50.885, abs=0.01)
    assert ps.source.startswith("imagej-info:")
