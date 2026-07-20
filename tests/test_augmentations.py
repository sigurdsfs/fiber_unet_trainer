from __future__ import annotations

import pytest

from fiberseg.augmentations import build_transform


def test_build_transform_rejects_gauss_noise_without_explicit_std_range() -> None:
    """GaussNoise's Albumentations default (std_range=(0.2, 0.44)) is calibrated
    for images with a much wider intensity range than this pipeline's [0, 1]
    normalized tiles; applied there it saturates the image with noise. Configs
    must set std_range explicitly."""
    with pytest.raises(ValueError, match="std_range"):
        build_transform([{"name": "GaussNoise", "p": 0.2}])


def test_build_transform_accepts_gauss_noise_with_explicit_std_range() -> None:
    transform = build_transform([{"name": "GaussNoise", "std_range": [0.02, 0.08], "p": 0.2}])
    assert transform is not None
