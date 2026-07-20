# predict_tiles.py
"""Tiled inference on a single large grayscale image using a trained checkpoint.

Reuses `_hw`, `_normalize_image`, `_read_gray` from `dataset.py` so preprocessing is
identical between training and inference. `load_predictor`, `predict_mask`, and
`save_mask` are also imported by `predict_all.py` to run the same inference over every
image in a config's `data.images_dir` without duplicating the tiling logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from .config import AppConfig, load_config
from .dataset import _hw, _normalize_image, _read_gray
from .lit_module import FiberSegmentationLitModule


def _make_model_input(tile: np.ndarray, image_channels: int, device: torch.device) -> torch.Tensor:
    """Add batch/channel dims to a normalized HxW tile and move it to `device`."""
    if image_channels == 1:
        tile = tile[None, :, :]
    elif image_channels == 3:
        tile = np.stack([tile, tile, tile], axis=0)
    else:
        raise ValueError(
            f"Unsupported image_channels={image_channels}. "
            "Use image_channels: 1 or image_channels: 3."
        )

    return torch.from_numpy(np.ascontiguousarray(tile)).float().unsqueeze(0).to(device)


def predict_mask(
    img: np.ndarray,
    model: FiberSegmentationLitModule,
    cfg: AppConfig,
    device: torch.device,
) -> np.ndarray:
    """Run tiled inference on a single normalized grayscale image, returning a 0/255 uint8 mask."""
    patch_h, patch_w = _hw(cfg.data.patch_size)
    stride_h, stride_w = _hw(cfg.data.stride or cfg.data.patch_size)

    H, W = img.shape[:2]

    prob = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(1, H - patch_h + 1), stride_h))
    xs = list(range(0, max(1, W - patch_w + 1), stride_w))

    if ys[-1] != max(0, H - patch_h):
        ys.append(max(0, H - patch_h))

    if xs[-1] != max(0, W - patch_w):
        xs.append(max(0, W - patch_w))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img[y:y+patch_h, x:x+patch_w]

                pad_h = patch_h - tile.shape[0]
                pad_w = patch_w - tile.shape[1]

                if pad_h or pad_w:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)), constant_values=0)

                x_t = _make_model_input(
                    tile=tile,
                    image_channels=cfg.data.image_channels,
                    device=device,
                )

                p = torch.sigmoid(model(x_t))[0, 0].cpu().numpy()
                p = p[:min(patch_h, H-y), :min(patch_w, W-x)]

                prob[y:y+p.shape[0], x:x+p.shape[1]] += p
                count[y:y+p.shape[0], x:x+p.shape[1]] += 1

    prob = prob / np.maximum(count, 1)

    return (prob > cfg.train.threshold).astype(np.uint8) * 255


def load_predictor(checkpoint: str, cfg: AppConfig) -> tuple[FiberSegmentationLitModule, torch.device]:
    """Load a checkpoint in eval mode onto CUDA if available, else CPU."""
    model = FiberSegmentationLitModule.load_from_checkpoint(
        checkpoint,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def save_mask(mask: np.ndarray, out: Path) -> None:
    """Write a 0/255 uint8 mask, using tifffile for .tif/.tiff and Pillow otherwise."""
    if out.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(out, mask)
    else:
        Image.fromarray(mask).save(out)


def main():
    """CLI entry point: predict a mask for a single `--image` and write it to `--out`."""
    parser = argparse.ArgumentParser(
        description="Run tiled prediction on one large grayscale image."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    img = _normalize_image(_read_gray(Path(args.image)))
    mask = predict_mask(img, model, cfg, device)

    save_mask(mask, Path(args.out))


if __name__ == "__main__":
    main()