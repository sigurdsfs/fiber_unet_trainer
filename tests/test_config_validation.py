from __future__ import annotations

from pathlib import Path

import pytest

from fiberseg.config import DataConfig, load_config
from fiberseg.dataset import find_pairs


def _write_config(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_config_rejects_invalid_split_fractions(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_split.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
  split:
    train: 0.8
    val: 0.2
    test: 0.2
""",
    )

    with pytest.raises(ValueError, match="sum to 1"):
        load_config(config_path)


def test_load_config_requires_images_and_masks_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_dirs.yaml"
    _write_config(
        config_path,
        """
data:
  split:
    train: 0.7
    val: 0.2
    test: 0.1
""",
    )

    with pytest.raises(ValueError, match="images_dir"):
        load_config(config_path)


def test_find_pairs_supports_custom_mask_pattern(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    (images_dir / "sample01.tif").write_bytes(b"fake-image")
    (masks_dir / "sample01_mask.png").write_bytes(b"fake-mask")

    cfg = DataConfig(images_dir=str(images_dir), masks_dir=str(masks_dir), mask_pattern="{stem}_mask.png")
    pairs = find_pairs(cfg)

    assert len(pairs) == 1
    assert pairs[0].mask_path.name == "sample01_mask.png"
