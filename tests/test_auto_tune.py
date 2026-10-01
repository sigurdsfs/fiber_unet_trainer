from pathlib import Path

import lightning
import numpy as np
import pytest
import torch
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold

from fiberseg.config import AppConfig, DataConfig, ModelConfig, load_config
from fiberseg.lit_module import FiberSegmentationLitModule
from fiberseg.predict_tiles import load_predictor
from fiberseg.tools import auto_tune
from fiberseg.tools.tune_hysteresis_threshold import hysteresis_counts
from fiberseg.train import _expand_sweep_configs
from fiberseg.tuned_thresholds import TUNED_KEY, apply_tuned

REPO = Path(__file__).resolve().parent.parent


def test_hysteresis_counts_matches_skimage():
    rng = np.random.default_rng(0)
    prob = ndi.gaussian_filter(rng.random((96, 96)), 2.0)
    prob = (prob - prob.min()) / (prob.max() - prob.min())
    fiber = ndi.gaussian_filter(rng.random((96, 96)), 2.0) > 0.5
    lows = np.array([0.2, 0.4, 0.55])
    highs = np.array([0.3, 0.5, 0.6, 0.8, 0.95])
    tp, fp, fn = hysteresis_counts(prob, fiber, lows, highs)
    for li, low in enumerate(lows):
        for hi, high in enumerate(highs):
            if low >= high:
                continue
            pred = apply_hysteresis_threshold(prob, low, high)
            assert tp[li, hi] == np.sum(pred & fiber)
            assert fp[li, hi] == np.sum(pred & ~fiber)
            assert fn[li, hi] == np.sum(~pred & fiber)


def _synthetic_images():
    """A bright 'fibre' line (prob 0.9) and a dimmer 'scratch' line (prob 0.6)."""
    prob = np.full((64, 64), 0.05, np.float32)
    fiber = np.zeros((64, 64), bool)
    prob[10:12, 5:60] = 0.9
    fiber[10:12, 5:60] = True
    prob[40:42, 5:60] = 0.6
    return [("img.tif", prob, fiber)]


def test_tune_modes_rejects_scratch_and_evaluates(monkeypatch):
    monkeypatch.setattr(auto_tune, "_split_images", lambda *a, **k: iter(_synthetic_images()))
    cfg = AppConfig(data=DataConfig(images_dir="x", masks_dir="y"))
    tuned = auto_tune.tune_modes(cfg, None, None, ["fixed", "hysteresis"])
    assert 0.6 <= tuned["modes"]["fixed"]["threshold"] < 0.9
    assert tuned["modes"]["hysteresis"]["high"] >= 0.6
    assert tuned["val"]["fixed"]["dice"] == pytest.approx(1.0, abs=1e-6)

    frames, pooled = auto_tune.evaluate_modes(cfg, None, None, tuned["modes"])
    assert set(frames) == {"fixed-untuned", "fixed", "hysteresis"}
    assert pooled["fixed"]["dice"] == pytest.approx(1.0, abs=1e-6)
    # The untuned 0.5 threshold keeps the scratch, so precision halves.
    assert pooled["fixed-untuned"]["precision"] == pytest.approx(0.5, abs=1e-6)
    assert {"image", "split", "dice", "tp", "fp", "fn"} <= set(frames["fixed"].columns)


def test_apply_tuned_sets_every_mode():
    cfg = AppConfig(data=DataConfig(images_dir="x", masks_dir="y"))
    applied = apply_tuned(cfg, {"modes": {
        "fixed": {"threshold": 0.81},
        "hysteresis": {"low": 0.3, "high": 0.9},
        "ridge": {"ridge_threshold": 0.2, "ridge_method": "sato", "ridge_sigmas": [1.0]},
    }})
    assert len(applied) == 3
    assert cfg.train.threshold == 0.81
    assert (cfg.inference.hysteresis_low, cfg.inference.hysteresis_high) == (0.3, 0.9)
    assert cfg.inference.ridge_threshold == 0.2
    assert cfg.inference.ridge_method == "sato"
    assert cfg.inference.threshold_mode == "fixed"
    assert apply_tuned(cfg, None) == []


def test_tuned_thresholds_roundtrip_through_checkpoint(tmp_path):
    cfg = AppConfig(
        data=DataConfig(images_dir="x", masks_dir="y"),
        model=ModelConfig(encoder_name="resnet18", encoder_weights=None, in_channels=1),
    )
    module = FiberSegmentationLitModule(cfg.model, cfg.train)
    ckpt = tmp_path / "best.ckpt"
    torch.save({
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }, ckpt)

    auto_tune.embed_in_checkpoint(ckpt, {"modes": {"fixed": {"threshold": 0.77}}})
    assert torch.load(ckpt, weights_only=False)[TUNED_KEY]["modes"]["fixed"]["threshold"] == 0.77

    load_predictor(str(ckpt), cfg)
    assert cfg.train.threshold == 0.77


def test_sweep_keeps_base_run_name_as_prefix():
    cfg = load_config(REPO / "configs" / "proxy" / "resnet34_baseline.yaml")
    cfg.mlflow.run_name = "baseline"
    names = [c.mlflow.run_name for c in _expand_sweep_configs(cfg)]
    assert names == ["baseline | seed=1", "baseline | seed=2", "baseline | seed=3"]


def test_invalid_auto_tune_mode_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data: {images_dir: x, masks_dir: y}\ntrain: {auto_tune_modes: [otsu]}\n")
    with pytest.raises(ValueError, match="auto_tune_modes"):
        load_config(p)
