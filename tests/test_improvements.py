"""Unit tests for the performance-improvement additions (see IMPROVEMENTS.md).

Covers the pure, deterministic pieces: channel standardization, the Gaussian tile
window, TTA transform invertibility, soft-clDice topology sensitivity, epoch-wise
metric micro-averaging, and the vectorized threshold-sweep counting.
"""
from __future__ import annotations

import numpy as np
import torch

from fiberseg.dataset import WeightedTileSampler, _apply_channel_norm
from fiberseg.lit_module import (
    _confusion_counts,
    _soft_cldice,
    _stats_from_counts,
)
from fiberseg.predict_tiles import _TTA_TRANSFORMS, _gaussian_window
from fiberseg.tools.tune_threshold import _counts_at


# --- channel standardization -------------------------------------------------

def test_minmax_normalization_is_identity():
    chw = np.random.rand(3, 8, 8).astype(np.float32)
    out = _apply_channel_norm(chw, "minmax")
    assert np.array_equal(out, chw)


def test_imagenet_normalization_3ch_uses_per_channel_stats():
    chw = np.stack([
        np.full((4, 4), 0.485, np.float32),
        np.full((4, 4), 0.456, np.float32),
        np.full((4, 4), 0.406, np.float32),
    ])
    out = _apply_channel_norm(chw, "imagenet")
    # Feeding exactly the ImageNet mean must map every channel to ~0.
    assert np.allclose(out, 0.0, atol=1e-5)


def test_imagenet_normalization_1ch():
    chw = np.full((1, 4, 4), 0.449, np.float32)
    out = _apply_channel_norm(chw, "imagenet")
    assert np.allclose(out, 0.0, atol=1e-4)


def test_unknown_normalization_raises():
    try:
        _apply_channel_norm(np.zeros((3, 2, 2), np.float32), "zscore")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown normalization mode")


def test_dataset_normalization_uses_provided_stats():
    chw = np.stack([
        np.full((2, 2), 0.5, np.float32),
        np.full((2, 2), 0.4, np.float32),
        np.full((2, 2), 0.3, np.float32),
    ])
    out = _apply_channel_norm(chw, "dataset", [0.5, 0.4, 0.3], [0.1, 0.1, 0.1])
    # feeding exactly the per-channel mean maps every channel to ~0.
    assert np.allclose(out, 0.0, atol=1e-5)


def test_dataset_normalization_requires_stats():
    try:
        _apply_channel_norm(np.zeros((1, 2, 2), np.float32), "dataset")
    except ValueError as e:
        assert "compute_dataset_stats" in str(e)
        return
    raise AssertionError("expected ValueError when dataset stats are missing")


def test_dataset_normalization_rejects_length_mismatch():
    try:
        _apply_channel_norm(np.zeros((3, 2, 2), np.float32), "dataset", [0.5], [0.1])
    except ValueError:
        return
    raise AssertionError("expected ValueError for stats length != channels")


# --- gaussian tile window ----------------------------------------------------

def test_gaussian_window_peaks_at_center_and_stays_positive():
    w = _gaussian_window(64, 64)
    assert w.shape == (64, 64)
    assert w.min() > 0.0
    assert w[32, 32] == w.max()
    # centre must dominate a corner so borders are down-weighted.
    assert w[32, 32] > 10 * w[0, 0]


# --- TTA invertibility -------------------------------------------------------

def test_all_tta_transforms_are_exact_inverses():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)  # asymmetric
    for i, (fwd, inv) in enumerate(_TTA_TRANSFORMS):
        back = np.ascontiguousarray(inv(np.ascontiguousarray(fwd(a))))
        assert np.array_equal(back, a), f"transform {i} is not an exact inverse"


# --- soft clDice -------------------------------------------------------------

def _logit(mask, p_fg=0.999, p_bg=0.001):
    fg = torch.log(torch.tensor(p_fg) / (1 - torch.tensor(p_fg)))
    bg = torch.log(torch.tensor(p_bg) / (1 - torch.tensor(p_bg)))
    return fg * mask + bg * (1 - mask)


def test_cldice_penalizes_broken_fiber():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 8, 2:14] = 1.0
    broken = target.clone()
    broken[0, 0, 8, 7:9] = 0.0  # sever the fiber: few pixels, big topology change

    cl_perfect = _soft_cldice(_logit(target), target, iters=5).item()
    cl_broken = _soft_cldice(_logit(broken), target, iters=5).item()
    assert cl_perfect > 0.95
    assert cl_broken < cl_perfect


def test_cldice_is_differentiable():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 8, 2:14] = 1.0
    x = _logit(target, 0.6, 0.4).clone().requires_grad_(True)
    loss = 1.0 - _soft_cldice(x, target, iters=3).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# --- epoch-wise metric micro-averaging --------------------------------------

def test_confusion_counts_and_stats_perfect_prediction():
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    tp, fp, fn = _confusion_counts(logits, target, 0.5)
    assert (tp, fp, fn) == (2.0, 0.0, 0.0)
    stats = _stats_from_counts(tp, fp, fn)
    assert abs(stats["dice"] - 1.0) < 1e-4
    assert abs(stats["tversky"] - 1.0) < 1e-4


def test_empty_tile_contributes_zero_counts():
    # An all-background tile predicted all-background must add nothing to the
    # epoch totals - the whole point of micro-averaging over per-tile ratios.
    logits = torch.full((1, 1, 4, 4), -10.0)
    target = torch.zeros((1, 1, 4, 4))
    assert _confusion_counts(logits, target, 0.5) == (0.0, 0.0, 0.0)


# --- weighted tile sampler (optional hard-negative mining) -------------------

def _sampler(n_pos=10, n_neg=90, **kw):
    is_pos = [True] * n_pos + [False] * n_neg
    kw.setdefault("negative_ratio", 1.0)
    kw.setdefault("hard_negative_fraction", 0.0)
    kw.setdefault("warmup_epochs", 0)
    kw.setdefault("seed", 42)
    return WeightedTileSampler(is_pos, **kw)


def test_sampler_includes_all_positives_every_epoch():
    s = _sampler(n_pos=10, n_neg=90, negative_ratio=1.0)
    for _ in range(3):
        drawn = set(iter(s))
        assert set(range(10)).issubset(drawn), "not all positives sampled"


def test_sampler_negative_count_follows_ratio():
    s = _sampler(n_pos=10, n_neg=90, negative_ratio=2.0)
    assert len(s) == 10 + 20  # all positives + 2x positives negatives
    drawn = list(iter(s))
    assert sum(1 for i in drawn if i >= 10) == 20


def test_sampler_negatives_capped_at_available():
    s = _sampler(n_pos=10, n_neg=5, negative_ratio=5.0)  # wants 50, only 5 exist
    assert len(s) == 10 + 5


def test_sampler_hard_negative_bias_after_warmup():
    s = _sampler(n_pos=10, n_neg=90, negative_ratio=2.0,
                 hard_negative_fraction=0.5, warmup_epochs=0)
    # tiles 10..19 are "hard" (high loss); the rest ordinary.
    diff = np.ones(100)
    diff[10:20] = 100.0
    s.update_difficulty(np.arange(100), diff)

    from collections import Counter
    counts = Counter()
    for _ in range(60):
        for i in iter(s):
            if i >= 10:
                counts[i] += 1
    hard_rate = sum(counts[i] for i in range(10, 20)) / 10
    other_rate = sum(counts[i] for i in range(20, 100)) / 80
    assert hard_rate > 3 * other_rate, "hard negatives should be drawn far more often"


def test_sampler_warmup_is_uniform():
    s = _sampler(n_pos=10, n_neg=90, negative_ratio=2.0,
                 hard_negative_fraction=1.0, warmup_epochs=5)
    diff = np.ones(100)
    diff[10:20] = 1000.0
    s.update_difficulty(np.arange(100), diff)
    # during warmup (epoch < 5) the hard tiles must NOT be preferentially drawn.
    from collections import Counter
    counts = Counter()
    for _ in range(5):  # all within warmup
        for i in iter(s):
            if i >= 10:
                counts[i] += 1
    hard_rate = sum(counts[i] for i in range(10, 20)) / 10
    other_rate = sum(counts[i] for i in range(20, 100)) / 80
    assert hard_rate < 2 * other_rate, "warmup draw should be roughly uniform"


# --- threshold sweep counting ------------------------------------------------

def test_counts_at_matches_naive_thresholding():
    rng = np.random.default_rng(0)
    prob = rng.random((32, 32)).astype(np.float32)
    gt = (rng.random((32, 32)) > 0.7).astype(np.uint8)
    thresholds = np.linspace(0.0, 1.0, 11)[1:-1]

    tp, fp, fn = _counts_at(prob, gt, thresholds)
    for i, t in enumerate(thresholds):
        pred = prob > t
        targ = gt > 0
        assert tp[i] == np.sum(pred & targ)
        assert fp[i] == np.sum(pred & ~targ)
        assert fn[i] == np.sum(~pred & targ)
