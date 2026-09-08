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


def test_load_config_rejects_negative_negative_ratio_final(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_ratio_final.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
  tile_sampling: weighted
  negative_ratio_final: -1.0
""",
    )

    with pytest.raises(ValueError, match="negative_ratio_final"):
        load_config(config_path)


def test_load_config_rejects_negative_ratio_anneal_epochs(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_anneal_epochs.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
  tile_sampling: weighted
  negative_ratio_anneal_epochs: -1
""",
    )

    with pytest.raises(ValueError, match="negative_ratio_anneal_epochs"):
        load_config(config_path)


def test_load_config_accepts_ratio_annealing_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "ratio_annealing.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
  tile_sampling: weighted
  negative_ratio: 1.0
  negative_ratio_final: 4.0
  negative_ratio_anneal_epochs: 10
""",
    )

    cfg = load_config(config_path)
    assert cfg.data.negative_ratio_final == 4.0
    assert cfg.data.negative_ratio_anneal_epochs == 10


def test_load_config_rejects_negative_focal_bce_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_focal_bce_weight.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
train:
  loss:
    focal_bce_weight: -1.0
""",
    )

    with pytest.raises(ValueError, match="focal_bce_weight"):
        load_config(config_path)


def test_load_config_rejects_out_of_range_focal_bce_alpha(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_focal_bce_alpha.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
train:
  loss:
    focal_bce_alpha: 1.5
""",
    )

    with pytest.raises(ValueError, match="focal_bce_alpha"):
        load_config(config_path)


def test_load_config_accepts_focal_bce_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "focal_bce.yaml"
    _write_config(
        config_path,
        """
data:
  images_dir: data/images
  masks_dir: data/masks
train:
  loss:
    name: focal_tversky
    focal_bce_weight: 1.0
    focal_bce_alpha: 0.25
    focal_bce_gamma: 2.0
""",
    )

    cfg = load_config(config_path)
    assert cfg.train.loss.focal_bce_weight == 1.0
    assert cfg.train.loss.focal_bce_alpha == 0.25
    assert cfg.train.loss.focal_bce_gamma == 2.0


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
