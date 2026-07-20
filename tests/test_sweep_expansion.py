from pathlib import Path

from fiberseg.config import load_config
from fiberseg.train import _expand_sweep_configs


def test_expand_sweep_configs_creates_one_config_per_combination(tmp_path: Path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        """
data:
  images_dir: ./data/images
  masks_dir: ./data/masks
  patch_size: 512
  stride: 512
model:
  encoder_name: resnet34
train:
  learning_rate: 0.0001
sweep:
  data.patch_size: [256, 512]
  train.learning_rate: [0.0001, 0.0002]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    configs = list(_expand_sweep_configs(cfg))

    assert len(configs) == 4
    assert configs[0].data.patch_size == 256
    assert configs[0].train.learning_rate == 0.0001
    assert configs[-1].data.patch_size == 512
    assert configs[-1].train.learning_rate == 0.0002
