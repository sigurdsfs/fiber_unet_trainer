"""Equivalence tests for the torch/GPU port of skimage's Frangi/Sato ridge filters.

The whole point of `enhance_ridges`' GPU path is that it is the SAME filter, just
evaluated with torch - a 46 MP Frangi costs ~60 s in skimage versus ~1 s on the GPU.
These tests pin that equivalence so the fast path can't silently drift into computing
something else. They run on the CPU too (torch backend, `device="cpu"`), so they stay
meaningful on a machine without CUDA.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix
from skimage.filters import frangi, sato

from fiberseg.tools.fiber_gap_repair import (
    _hessian_torch,
    _ridge_torch,
    enhance_ridges,
)

SIGMAS = (1, 2, 3)


def _fiber_image(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """A synthetic probability-map-like image: thin bright ridges plus noise."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w)) * 0.15
    for _ in range(12):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        angle = float(rng.random() * np.pi)
        for t in range(int(rng.integers(40, 120))):
            yy = int(y + t * np.sin(angle))
            xx = int(x + t * np.cos(angle))
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx] = 1.0
    return ndi.gaussian_filter(img, 1.0)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_hessian_matches_skimage(sigma):
    img = _fiber_image()
    got = _hessian_torch(torch.from_numpy(img)[None, None], sigma)
    ref = hessian_matrix(img, sigma, mode="reflect", use_gaussian_derivatives=True)
    for g, r in zip(got, ref):
        scale = np.abs(r).max() + 1e-12
        assert np.abs(g[0, 0].numpy() - r).max() / scale < 1e-10


@pytest.mark.parametrize("method,reference", [("frangi", frangi), ("sato", sato)])
def test_ridge_torch_matches_skimage_float64(method, reference):
    img = _fiber_image()
    got = _ridge_torch(
        torch.from_numpy(img)[None, None], method=method, sigmas=SIGMAS
    )[0, 0].numpy()
    ref = reference(img, sigmas=SIGMAS, black_ridges=False)
    scale = np.abs(ref).max() + 1e-12
    assert np.abs(got - ref).max() / scale < 1e-10


def test_enhance_ridges_cpu_backend_matches_skimage():
    # device="cpu" takes the skimage path; assert the public entry point is stable.
    img = _fiber_image()
    got = enhance_ridges(img, method="frangi", sigmas=SIGMAS, device="cpu")
    ref = frangi(img.astype(np.float64), sigmas=SIGMAS, black_ridges=False)
    assert np.allclose(got, ref)


@pytest.mark.parametrize("method", ["frangi", "sato"])
def test_float32_is_close_enough_for_thresholding(method):
    """float32 is the GPU default; it must not move the thresholded mask."""
    img = _fiber_image()
    ref = _ridge_torch(
        torch.from_numpy(img)[None, None], method=method, sigmas=SIGMAS
    )[0, 0].numpy()
    got = _ridge_torch(
        torch.from_numpy(img).float()[None, None], method=method, sigmas=SIGMAS
    )[0, 0].numpy()

    scale = np.abs(ref).max() + 1e-12
    assert np.abs(got - ref).max() / scale < 1e-5
    # what actually matters downstream: the binarized mask must be identical.
    for q in (0.99, 0.999):
        a = ref > np.quantile(ref, q)
        b = got > np.quantile(got, q)
        union = (a | b).sum()
        assert union == 0 or (a & b).sum() / union > 0.999


@pytest.mark.parametrize("size", [16, 32, 64])
def test_small_images_smaller_than_kernel_radius(size):
    """sigma<=1 uses truncate=100 -> radius 71, which exceeds a small image.

    scipy's reflect padding repeats the mirror pattern for radii larger than the
    axis; a naive single flip-and-concat silently produces a too-small array (this
    used to blow up with a shape mismatch on 64px tiles).
    """
    img = _fiber_image(size, size, seed=size)
    got = _ridge_torch(
        torch.from_numpy(img)[None, None], method="frangi", sigmas=SIGMAS
    )[0, 0].numpy()
    ref = frangi(img, sigmas=SIGMAS, black_ridges=False)
    assert got.shape == img.shape
    scale = np.abs(ref).max() + 1e-12
    assert np.abs(got - ref).max() / scale < 1e-10


def test_non_square_image():
    img = _fiber_image(48, 130, seed=7)
    got = _ridge_torch(
        torch.from_numpy(img)[None, None], method="frangi", sigmas=SIGMAS
    )[0, 0].numpy()
    ref = frangi(img, sigmas=SIGMAS, black_ridges=False)
    assert got.shape == img.shape
    assert np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12) < 1e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("method", ["frangi", "sato"])
def test_cuda_matches_cpu(method):
    img = _fiber_image()
    gpu = enhance_ridges(img, method=method, sigmas=SIGMAS, device="cuda")
    cpu = enhance_ridges(img, method=method, sigmas=SIGMAS, device="cpu")
    scale = np.abs(cpu).max() + 1e-12
    assert np.abs(gpu - cpu).max() / scale < 1e-5
