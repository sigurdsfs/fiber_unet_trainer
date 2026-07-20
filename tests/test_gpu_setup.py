from __future__ import annotations

from fiberseg.config import TrainConfig
from fiberseg.train import resolve_trainer_settings


def test_resolve_trainer_settings_prefers_gpu_when_available() -> None:
    cfg = TrainConfig(accelerator="auto", devices="auto", precision="32-true")

    accelerator, devices, precision = resolve_trainer_settings(cfg, cuda_available=True)

    assert accelerator == "gpu"
    assert devices == "auto"
    assert precision == "16-mixed"


def test_resolve_trainer_settings_requires_gpu_when_requested() -> None:
    cfg = TrainConfig(accelerator="gpu", devices="auto", precision="16-mixed")

    try:
        resolve_trainer_settings(cfg, cuda_available=False)
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when GPU was requested but unavailable")
