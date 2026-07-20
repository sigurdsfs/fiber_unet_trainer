import torch

from fiberseg.config import load_config
from fiberseg.models import create_model


def main():
    cfg = load_config("configs/cnn_micronet_resnet50.yaml")

    print("Creating model...")
    model = create_model(cfg.model)
    model.eval()

    print("Model created.")
    print(f"Input channels: {cfg.model.in_channels}")
    print(f"Classes: {cfg.model.classes}")

    x = torch.randn(2, cfg.model.in_channels, 512, 512)

    print("Running forward pass...")

    with torch.no_grad():
        y = model(x)

    print("Output type:", type(y))

    if hasattr(y, "shape"):
        print("Output shape:", y.shape)
    else:
        raise TypeError(f"Model output is not a tensor. Got: {type(y)}")

    expected_shape = (2, cfg.model.classes, 512, 512)

    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape. Expected {expected_shape}, got {tuple(y.shape)}"
        )

    print("MicroNet forward test passed.")


if __name__ == "__main__":
    main()