from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fiberseg.config import DataConfig
from fiberseg.dataset import TiledSegmentationDataset
from fiberseg.train import _log_filtered_tile_stats


def test_train_dataset_reports_filtered_patches(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    image = np.zeros((8, 8), dtype=np.uint8)
    image[:4, :4] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4, :4] = 1

    Image.fromarray(image, mode="L").save(images_dir / "sample.png")
    Image.fromarray(mask, mode="L").save(masks_dir / "sample_mask.png")

    cfg = DataConfig(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        image_glob="*.png",
        mask_pattern="{stem}_mask.png",
        patch_size=4,
        stride=4,
        min_foreground_fraction=0.5,
        keep_empty_probability=0.0,
        seed=123,
    )

    dataset = TiledSegmentationDataset(cfg, split="train")

    assert len(dataset.tiles) == 1
    assert dataset.filtered_tiles_count == 3
    assert dataset.filtered_tiles_fraction == 0.75


class _DummyLogger:
    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, float]] = []
        self.logged_steps: list[int | None] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged_metrics.append(dict(metrics))
        self.logged_steps.append(step)


def test_log_filtered_tile_stats_uses_logger_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DataConfig(images_dir="/tmp/images", masks_dir="/tmp/masks")

    class _DummyDatamodule:
        filtered_tiles_count = 2
        filtered_tiles_fraction = 0.5

        def __init__(self) -> None:
            self.train_ds = type("TrainDS", (), {"tiles": [1, 2, 3]})()

    logger = _DummyLogger()
    monkeypatch.setattr("fiberseg.train.mlflow.set_tracking_uri", lambda *_args, **_kwargs: None)

    _log_filtered_tile_stats(cfg, _DummyDatamodule(), logger)

    assert logger.logged_metrics[0]["filtered_tiles_count"] == 2.0
    assert logger.logged_metrics[0]["filtered_tiles_fraction"] == 0.5
