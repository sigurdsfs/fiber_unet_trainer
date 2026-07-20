from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fiberseg.config import load_config
from fiberseg.dataset import FiberDataModule
from fiberseg.lit_module import FiberSegmentationLitModule


def find_fiber_sample(dataset, max_scan: int = 5000):
    for idx in range(min(len(dataset), max_scan)):
        img, mask = dataset[idx]
        fg = int((mask > 0.5).sum().item())

        if fg > 0:
            return idx, img, mask

    raise RuntimeError("Could not find a fiber-containing sample.")


def make_input_variants(img: torch.Tensor):
    """
    img is expected as [C, H, W], usually 3-channel duplicated grayscale.
    Returns alternative preprocessing variants.
    """
    variants = {}

    x = img.clone().float()

    # Current pipeline.
    variants["current_0_1"] = x

    # Inverted contrast, in case fibers are opposite polarity compared with pretraining.
    variants["inverted_0_1"] = 1.0 - x

    # Per-image standardization.
    mean = x.mean()
    std = x.std().clamp_min(1e-6)
    variants["per_image_standardized"] = (x - mean) / std

    # ImageNet-style normalization on duplicated grayscale.
    # This is often what RGB pretrained encoders expect.
    if x.shape[0] == 3:
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        variants["imagenet_normalized"] = (x - imagenet_mean) / imagenet_std

    return variants


def summarize_output(name: str, logits: torch.Tensor, mask: torch.Tensor):
    probs = torch.sigmoid(logits)

    print(f"\n=== {name} ===")
    print(f"logits min:  {float(logits.min()):.6f}")
    print(f"logits mean: {float(logits.mean()):.6f}")
    print(f"logits max:  {float(logits.max()):.6f}")
    print(f"prob min:    {float(probs.min()):.6f}")
    print(f"prob mean:   {float(probs.mean()):.6f}")
    print(f"prob max:    {float(probs.max()):.6f}")
    print(f"pred px >0.1: {int((probs > 0.1).sum().item())}")
    print(f"pred px >0.2: {int((probs > 0.2).sum().item())}")
    print(f"pred px >0.5: {int((probs > 0.5).sum().item())}")
    print(f"mask px:      {int((mask > 0.5).sum().item())}")

    return probs.detach().cpu()[0, 0].numpy()


def image_for_display(img: torch.Tensor):
    x = img.detach().cpu().float()

    if x.ndim == 3 and x.shape[0] >= 3:
        arr = x[:3].permute(1, 2, 0).numpy()
    elif x.ndim == 3:
        arr = x[0].numpy()
    else:
        arr = x.numpy()

    arr = arr.astype(np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))

    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    return arr


def save_debug_figure(out_path: Path, img, mask, prob_maps: dict[str, np.ndarray]):
    n = 2 + len(prob_maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(image_for_display(img), cmap="gray")
    axes[0].set_title("input")

    axes[1].imshow(mask.detach().cpu().squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("mask")

    for ax, (name, prob) in zip(axes[2:], prob_maps.items()):
        ax.imshow(prob, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default="debug_probability_response.png")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dm = FiberDataModule(cfg.data, cfg.augmentations)

    if args.split == "test":
        dm.setup("test")
        dataset = dm.test_ds
    else:
        dm.setup("fit")
        dataset = dm.train_ds if args.split == "train" else dm.val_ds

    idx, img, mask = find_fiber_sample(dataset)

    print("Using dataset index:", idx)
    print("Image tensor shape:", tuple(img.shape))
    print("Mask tensor shape:", tuple(mask.shape))
    print("Image min/mean/max:", float(img.min()), float(img.mean()), float(img.max()))
    print("Mask pixels:", int((mask > 0.5).sum().item()))

    if args.checkpoint:
        print("Loading checkpoint:", args.checkpoint)
        model = FiberSegmentationLitModule.load_from_checkpoint(
            args.checkpoint,
            model_cfg=cfg.model,
            train_cfg=cfg.train,
            map_location="cpu",
        )
    else:
        print("Creating fresh untrained model.")
        model = FiberSegmentationLitModule(cfg.model, cfg.train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prob_maps = {}

    variants = make_input_variants(img)

    with torch.no_grad():
        for name, variant in variants.items():
            x = variant.unsqueeze(0).to(device)
            logits = model(x)
            prob = summarize_output(name, logits.detach().cpu(), mask)
            prob_maps[name] = prob

    save_debug_figure(
        out_path=Path(args.out),
        img=img,
        mask=mask,
        prob_maps=prob_maps,
    )

    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()