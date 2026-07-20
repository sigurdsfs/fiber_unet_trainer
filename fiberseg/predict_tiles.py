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
from .dataset import _apply_channel_norm, _hw, _normalize_image, _read_gray
from .lit_module import FiberSegmentationLitModule

# 8 dihedral (flip/rotate) transforms for test-time augmentation. Each entry is
# (forward, inverse): forward maps a tile into an augmented view, inverse maps a
# prediction on that view back to the original orientation. inverse(forward(x))
# must be identity so probabilities align before averaging.
_TTA_TRANSFORMS = [
    (lambda a: a, lambda a: a),
    (lambda a: np.rot90(a, 1), lambda a: np.rot90(a, -1)),
    (lambda a: np.rot90(a, 2), lambda a: np.rot90(a, -2)),
    (lambda a: np.rot90(a, 3), lambda a: np.rot90(a, -3)),
    (lambda a: np.fliplr(a), lambda a: np.fliplr(a)),
    (lambda a: np.flipud(a), lambda a: np.flipud(a)),
    (lambda a: np.rot90(np.fliplr(a), 1), lambda a: np.fliplr(np.rot90(a, -1))),
    (lambda a: np.rot90(np.fliplr(a), 3), lambda a: np.fliplr(np.rot90(a, -3))),
]


def _gaussian_window(h: int, w: int, sigma_scale: float = 0.125) -> np.ndarray:
    """2D Gaussian weight map (peak 1.0 at center) for blending overlapping tiles.

    Down-weights a tile's unreliable border pixels relative to its center so seams
    between overlapping tiles are smoothed - the nnU-Net sliding-window scheme.
    sigma is a fraction of the tile size; the small floor keeps edge weights > 0.
    """
    yy = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    xx = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    gy = np.exp(-(yy ** 2) / (2.0 * (sigma_scale * h) ** 2))
    gx = np.exp(-(xx ** 2) / (2.0 * (sigma_scale * w) ** 2))
    win = np.outer(gy, gx).astype(np.float32)
    return np.maximum(win, 1e-4)


def _make_model_input(
    tile: np.ndarray,
    image_channels: int,
    normalization: str,
    device: torch.device,
    norm_mean: list[float] | None = None,
    norm_std: list[float] | None = None,
) -> torch.Tensor:
    """Add batch/channel dims to a normalized HxW tile, standardize it, move to `device`.

    Applies the same `_apply_channel_norm` as dataset.__getitem__ (including the same
    norm_mean/norm_std for "dataset" mode) so the tensor fed to the model at inference
    is distributed identically to training tensors.
    """
    if image_channels == 1:
        tile = tile[None, :, :]
    elif image_channels == 3:
        tile = np.stack([tile, tile, tile], axis=0)
    else:
        raise ValueError(
            f"Unsupported image_channels={image_channels}. "
            "Use image_channels: 1 or image_channels: 3."
        )

    tile = _apply_channel_norm(tile.astype(np.float32), normalization, norm_mean, norm_std)
    return torch.from_numpy(np.ascontiguousarray(tile)).float().unsqueeze(0).to(device)


def _infer_tile_prob(
    tile: np.ndarray,
    model: FiberSegmentationLitModule,
    cfg: AppConfig,
    device: torch.device,
) -> np.ndarray:
    """Sigmoid probability for one full patch-sized tile, optionally TTA-averaged."""
    def _input(t):
        return _make_model_input(
            t,
            cfg.data.image_channels,
            cfg.data.image_normalization,
            device,
            cfg.data.norm_mean,
            cfg.data.norm_std,
        )

    if not cfg.inference.tta:
        return torch.sigmoid(model(_input(tile)))[0, 0].cpu().numpy()

    acc = np.zeros(tile.shape, dtype=np.float32)
    for forward, inverse in _TTA_TRANSFORMS:
        aug = np.ascontiguousarray(forward(tile))
        p_aug = torch.sigmoid(model(_input(aug)))[0, 0].cpu().numpy()
        acc += np.ascontiguousarray(inverse(p_aug))
    return acc / len(_TTA_TRANSFORMS)


def predict_prob(
    img: np.ndarray,
    model: FiberSegmentationLitModule,
    cfg: AppConfig,
    device: torch.device,
) -> np.ndarray:
    """Run tiled inference on a normalized grayscale image, returning a float32 [0,1]
    probability map (before thresholding).

    Edge tiles are reflect-padded (no artificial black border) and overlapping tiles
    are blended with a Gaussian window when cfg.inference.tile_blend == "gaussian".
    """
    patch_h, patch_w = _hw(cfg.data.patch_size)
    stride_h, stride_w = _hw(cfg.data.stride or cfg.data.patch_size)

    H, W = img.shape[:2]

    prob = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    if cfg.inference.tile_blend == "gaussian":
        window = _gaussian_window(patch_h, patch_w)
    elif cfg.inference.tile_blend == "uniform":
        window = np.ones((patch_h, patch_w), dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported tile_blend={cfg.inference.tile_blend!r}. Use 'gaussian' or 'uniform'."
        )

    pad_mode = "reflect" if cfg.inference.reflect_pad else "constant"

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
                    # reflect needs >=2 px along an axis; fall back to edge padding
                    # for degenerate 1px tiles so this never raises.
                    mode = pad_mode
                    if mode == "reflect" and (tile.shape[0] < 2 or tile.shape[1] < 2):
                        mode = "edge"
                    kwargs = {"constant_values": 0} if mode == "constant" else {}
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode=mode, **kwargs)

                p = _infer_tile_prob(tile, model, cfg, device)

                valid_h = min(patch_h, H - y)
                valid_w = min(patch_w, W - x)
                p = p[:valid_h, :valid_w]
                win = window[:valid_h, :valid_w]

                prob[y:y+valid_h, x:x+valid_w] += p * win
                weight[y:y+valid_h, x:x+valid_w] += win

    return prob / np.maximum(weight, 1e-8)


def predict_mask(
    img: np.ndarray,
    model: FiberSegmentationLitModule,
    cfg: AppConfig,
    device: torch.device,
) -> np.ndarray:
    """Run tiled inference on a single normalized grayscale image, returning a 0/255 uint8 mask.

    Thin wrapper over `predict_prob` that thresholds at `cfg.train.threshold`.
    """
    prob = predict_prob(img, model, cfg, device)
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