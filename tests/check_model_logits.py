from __future__ import annotations

import torch

from fiberseg.config import load_config
from fiberseg.models import create_model


def main():
    cfg = load_config("configs/cnn_micronet_resnet50.yaml")

    model = create_model(cfg.model)
    model.eval()

    print("\n=== Model ===")
    print(model.__class__)

    if hasattr(model, "segmentation_head"):
        print("\n=== Segmentation head ===")
        print(model.segmentation_head)

    x = torch.randn(2, cfg.model.in_channels, 512, 512)

    with torch.no_grad():
        y = model(x)

    print("\n=== Output ===")
    print("shape:", tuple(y.shape))
    print("min:", float(y.min()))
    print("mean:", float(y.mean()))
    print("max:", float(y.max()))

    if float(y.min()) >= 0.0 and float(y.max()) <= 1.0:
        print("\nWARNING: Output still looks like probabilities in [0, 1].")
        print("The model may still have an output activation.")
    else:
        print("\nOK: Output looks like raw logits.")


if __name__ == "__main__":
    main()