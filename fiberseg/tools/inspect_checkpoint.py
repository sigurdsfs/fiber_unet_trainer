# inspect_checkpoint.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import load_config
from ..dataset import FiberDataModule
from ..lit_module import FiberSegmentationLitModule


def image_to_numpy(img: torch.Tensor):
    img = img.detach().cpu().float()

    if img.ndim == 3 and img.shape[0] == 1:
        arr = img[0].numpy()
    elif img.ndim == 3 and img.shape[0] >= 3:
        arr = img[:3].permute(1, 2, 0).numpy()
    elif img.ndim == 2:
        arr = img.numpy()
    else:
        arr = img.squeeze().numpy()

    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)

    return arr


def mask_to_numpy(mask: torch.Tensor):
    mask = mask.detach().cpu().float().squeeze().numpy()
    return (mask > 0.5).astype(np.float32)


def make_overlay(img_np: np.ndarray, pred_np: np.ndarray):
    pred_bool = pred_np > 0.5

    if img_np.ndim == 2:
        base = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        base = img_np.copy()

    overlay = base.copy()
    overlay[pred_bool, 0] = 1.0
    overlay[pred_bool, 1] *= 0.4
    overlay[pred_bool, 2] *= 0.4

    return overlay


def save_prediction_panel(
    img: torch.Tensor,
    mask: torch.Tensor,
    prob: torch.Tensor,
    pred: torch.Tensor,
    out_path: Path,
    threshold: float,
):
    img_np = image_to_numpy(img)
    mask_np = mask_to_numpy(mask)
    prob_np = prob.squeeze().numpy()
    pred_np = pred.squeeze().numpy()
    overlay_np = make_overlay(img_np, pred_np)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    axes[0].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
    axes[0].set_title("Input")

    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    axes[2].imshow(prob_np, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Probability")

    axes[3].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction > {threshold}")

    axes[4].imshow(overlay_np)
    axes[4].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visual inspection of a trained segmentation checkpoint."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt checkpoint file.")
    parser.add_argument(
        "--out-dir",
        default="inspection_outputs",
        help="Directory for saved prediction panels.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to inspect.",
    )
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = cfg.train.threshold if args.threshold is None else args.threshold

    datamodule = FiberDataModule(cfg.data, cfg.augmentations)
    datamodule.setup("fit")
    datamodule.setup("test")

    if args.split == "train":
        dataset = datamodule.train_ds
    elif args.split == "val":
        dataset = datamodule.val_ds
    else:
        dataset = datamodule.test_ds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FiberSegmentationLitModule.load_from_checkpoint(
        args.checkpoint,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    n = min(args.max_images, len(dataset))

    with torch.no_grad():
        for i in range(n):
            img, mask = dataset[i]
            logits = model(img.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].detach().cpu()
            pred = (prob > threshold).float()

            out_path = out_dir / f"{args.split}_sample_{i:03d}.png"
            save_prediction_panel(
                img=img,
                mask=mask,
                prob=prob,
                pred=pred,
                out_path=out_path,
                threshold=threshold,
            )

    print(f"Saved {n} prediction panels to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()