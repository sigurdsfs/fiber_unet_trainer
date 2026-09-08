"""Tiled-inference tests: batching and GPU-side TTA must not change the output.

`predict_prob` batches tiles through the model and blends them in a device-resident
accumulator. Both are pure performance changes, so the guarantee that matters is that
the probability map is invariant to `inference.batch_size` and that the torch TTA
transforms are the same eight dihedral ops (and exact inverses) as the numpy ones.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fiberseg.config import AppConfig, DataConfig
from fiberseg.predict_tiles import (
    _TTA_TRANSFORMS,
    _TTA_TRANSFORMS_TORCH,
    predict_prob,
)


class _StripeModel(torch.nn.Module):
    """Deterministic, position-sensitive stand-in for a real segmentation net.

    Not symmetric under flips/rotations, so a TTA bug cannot cancel out.
    """

    def forward(self, x):
        n, _, h, w = x.shape
        ramp_y = torch.linspace(-2, 2, h, device=x.device).view(1, 1, h, 1)
        ramp_x = torch.linspace(-1, 3, w, device=x.device).view(1, 1, 1, w)
        return x.mean(dim=1, keepdim=True) * 3.0 + ramp_y + 0.5 * ramp_x


def _cfg(batch_size: int, tta: bool = False, patch=32, stride=16) -> AppConfig:
    cfg = AppConfig(data=DataConfig(images_dir="x", masks_dir="y"))
    cfg.data.patch_size = patch
    cfg.data.stride = stride
    cfg.data.image_channels = 1
    cfg.data.image_normalization = "minmax"
    cfg.inference.batch_size = batch_size
    cfg.inference.tta = tta
    return cfg


def _image(h=70, w=90, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w)).astype(np.float32)


# --- torch TTA transforms -----------------------------------------------------

def test_torch_tta_transforms_are_exact_inverses():
    a = torch.arange(2 * 1 * 5 * 7, dtype=torch.float32).reshape(2, 1, 5, 7)
    for i, (fwd, inv) in enumerate(_TTA_TRANSFORMS_TORCH):
        back = inv(fwd(a))
        assert torch.equal(back, a), f"torch transform {i} is not an exact inverse"


def test_torch_tta_matches_numpy_tta():
    """The torch variants must be the SAME eight dihedral ops as the numpy ones."""
    assert len(_TTA_TRANSFORMS_TORCH) == len(_TTA_TRANSFORMS)
    a = np.arange(35, dtype=np.float32).reshape(5, 7)
    t = torch.from_numpy(a)[None, None]
    for i, ((np_fwd, _), (t_fwd, _)) in enumerate(
        zip(_TTA_TRANSFORMS, _TTA_TRANSFORMS_TORCH)
    ):
        expected = np.ascontiguousarray(np_fwd(a))
        got = t_fwd(t)[0, 0].numpy()
        assert np.array_equal(got, expected), f"transform {i} differs from numpy"


# --- batching invariance ------------------------------------------------------

def test_predict_prob_is_invariant_to_batch_size():
    img = _image()
    model = _StripeModel().eval()
    device = torch.device("cpu")
    ref = predict_prob(img, model, _cfg(1), device)
    for bs in (2, 3, 8, 64):
        got = predict_prob(img, model, _cfg(bs), device)
        assert got.shape == img.shape
        assert np.allclose(ref, got, atol=1e-6), f"batch_size={bs} changed the output"


def test_predict_prob_is_invariant_to_batch_size_with_tta():
    img = _image()
    model = _StripeModel().eval()
    device = torch.device("cpu")
    ref = predict_prob(img, model, _cfg(1, tta=True), device)
    for bs in (3, 16):
        got = predict_prob(img, model, _cfg(bs, tta=True), device)
        assert np.allclose(ref, got, atol=1e-6), f"TTA batch_size={bs} changed output"


def test_predict_prob_covers_whole_image_and_is_a_probability():
    """Edge tiles are snapped to the border, so every pixel must be written."""
    img = _image(h=71, w=93)
    out = predict_prob(img, _StripeModel().eval(), _cfg(4), torch.device("cpu"))
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_predict_prob_handles_image_smaller_than_patch():
    img = _image(h=20, w=13)  # smaller than patch_size=32 -> padded edge tiles
    out = predict_prob(img, _StripeModel().eval(), _cfg(4), torch.device("cpu"))
    assert out.shape == img.shape
    assert np.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batching_does_not_move_the_mask_on_cuda():
    """On CUDA the comparison must be at mask level, not bitwise.

    cuDNN picks different convolution algorithms per batch shape, and with TF32
    enabled (the default on Ampere) that shifts individual probabilities by up to
    ~5e-2. Those shifts land away from the decision boundary - the binarized mask is
    unchanged - but it does mean batch_size is not a bitwise no-op on GPU, which the
    CPU tests above cannot observe.
    """
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    ).cuda().eval()
    img = _image(h=600, w=700)
    device = torch.device("cuda")

    ref = predict_prob(img, model, _cfg(1), device)
    got = predict_prob(img, model, _cfg(8), device)

    assert np.abs(ref - got).max() < 0.1, "probabilities moved far more than TF32 noise"
    a, b = ref > 0.5, got > 0.5
    union = (a | b).sum()
    assert union == 0 or (a & b).sum() / union > 0.9999
