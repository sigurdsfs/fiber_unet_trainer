from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fiberseg.augmentations import build_transform
from fiberseg.config import DataConfig
from fiberseg.dataset import TiledSegmentationDataset, _normalize_image, _read_gray


def _write_pair(
    images_dir: Path, masks_dir: Path, image: np.ndarray, mask: np.ndarray, stem: str
) -> None:
    Image.fromarray(image, mode="L").save(images_dir / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}_mask.png")


def test_tile_normalization_matches_whole_image_normalization(tmp_path: Path) -> None:
    """A tile's pixel values must come from normalizing the whole source image,
    not from normalizing the tile crop on its own - per-tile normalization gives
    every tile its own contrast stretch and diverges from predict_tiles.py, which
    normalizes the full image once before tiling."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    image = rng.integers(20, 40, size=(64, 64), dtype=np.uint8)  # dim, low-contrast background
    image[0:16, 0:16] = 220  # one bright patch, far from the probe tile below
    mask = np.zeros((64, 64), dtype=np.uint8)

    _write_pair(images_dir, masks_dir, image, mask, "sample")

    cfg = DataConfig(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        image_glob="*.png",
        mask_pattern="{stem}_mask.png",
        patch_size=32,
        stride=32,
        seed=0,
    )

    dataset = TiledSegmentationDataset(cfg, split="train")

    # Pick a tile that does not overlap the bright patch, so per-tile vs
    # per-image normalization actually give different answers.
    probe = next(t for t in dataset.tiles if t.y >= 32 and t.x >= 32)
    idx = dataset.tiles.index(probe)

    img_tensor, _ = dataset[idx]
    actual = img_tensor.squeeze(0).numpy()

    expected_whole = _normalize_image(_read_gray(images_dir / "sample.png"))
    expected = expected_whole[probe.y:probe.y + probe.h, probe.x:probe.x + probe.w]

    np.testing.assert_allclose(actual, expected, atol=1e-6)

    # Confirm per-tile normalization would have given a different result here -
    # otherwise this test would not actually exercise the bug it guards against.
    raw = _read_gray(images_dir / "sample.png")
    per_tile = _normalize_image(raw[probe.y:probe.y + probe.h, probe.x:probe.x + probe.w].copy())
    assert not np.allclose(actual, per_tile, atol=1e-3)


def test_augmentation_output_is_clipped_to_unit_range(tmp_path: Path) -> None:
    """Even an aggressive intensity augmentation must not push pixel values
    outside [0, 1] - the final clip after augmentation must always hold."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    image = np.full((32, 32), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    _write_pair(images_dir, masks_dir, image, mask, "sample")

    cfg = DataConfig(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        image_glob="*.png",
        mask_pattern="{stem}_mask.png",
        patch_size=32,
        stride=32,
        seed=0,
    )

    transform = build_transform(
        [
            {
                "name": "RandomBrightnessContrast",
                "brightness_limit": [1.0, 1.0],
                "contrast_limit": [1.0, 1.0],
                "p": 1.0,
            },
            {"name": "GaussNoise", "std_range": [0.4, 0.4], "p": 1.0},
        ]
    )

    dataset = TiledSegmentationDataset(cfg, split="train", augmentations=transform)

    img_tensor, _ = dataset[0]

    assert float(img_tensor.min()) >= 0.0
    assert float(img_tensor.max()) <= 1.0
