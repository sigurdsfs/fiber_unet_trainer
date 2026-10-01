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
from .tuned_thresholds import apply_tuned

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


# Torch equivalents of _TTA_TRANSFORMS, applied to the trailing (H, W) dims so they
# work on a whole (N, C, H, W) batch at once. Keeping TTA on the GPU avoids the
# 8x host<->device round trip and the ~24 full-tile numpy copies per tile that the
# numpy path costs; `tests/test_improvements.py` pins that both sets agree.
_TTA_TRANSFORMS_TORCH = [
    (lambda t: t, lambda t: t),
    (lambda t: torch.rot90(t, 1, (-2, -1)), lambda t: torch.rot90(t, -1, (-2, -1))),
    (lambda t: torch.rot90(t, 2, (-2, -1)), lambda t: torch.rot90(t, -2, (-2, -1))),
    (lambda t: torch.rot90(t, 3, (-2, -1)), lambda t: torch.rot90(t, -3, (-2, -1))),
    (lambda t: torch.flip(t, (-1,)), lambda t: torch.flip(t, (-1,))),
    (lambda t: torch.flip(t, (-2,)), lambda t: torch.flip(t, (-2,))),
    (lambda t: torch.rot90(torch.flip(t, (-1,)), 1, (-2, -1)),
     lambda t: torch.flip(torch.rot90(t, -1, (-2, -1)), (-1,))),
    (lambda t: torch.rot90(torch.flip(t, (-1,)), 3, (-2, -1)),
     lambda t: torch.flip(torch.rot90(t, -3, (-2, -1)), (-1,))),
]


def _infer_batch_prob(
    batch: torch.Tensor,
    model: FiberSegmentationLitModule,
    cfg: AppConfig,
) -> torch.Tensor:
    """Sigmoid probabilities for a (N, C, H, W) batch of tiles, optionally TTA-averaged.

    Returns an (N, H, W) tensor that stays on the model's device - the caller blends it
    into a device-resident accumulator, so a whole image costs one device->host copy
    instead of one per tile.
    """
    if not cfg.inference.tta:
        return torch.sigmoid(model(batch))[:, 0]

    acc = torch.zeros(
        (batch.shape[0], batch.shape[2], batch.shape[3]),
        dtype=torch.float32,
        device=batch.device,
    )
    for forward, inverse in _TTA_TRANSFORMS_TORCH:
        out = torch.sigmoid(model(forward(batch).contiguous()))[:, 0]
        acc += inverse(out)
    return acc / len(_TTA_TRANSFORMS_TORCH)


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

    # Accumulate on the model's device: at 46 MP these two buffers are ~370 MB total,
    # and keeping them here means the per-tile blend is a GPU op and the whole image
    # costs a single device->host transfer at the end.
    prob = torch.zeros((H, W), dtype=torch.float32, device=device)
    weight = torch.zeros((H, W), dtype=torch.float32, device=device)

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

    window_t = torch.from_numpy(window).to(device)
    coords = [(y, x) for y in ys for x in xs]
    batch_size = max(1, int(cfg.inference.batch_size))

    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            chunk = coords[start:start + batch_size]

            inputs = []
            for y, x in chunk:
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

                # _make_model_input returns (1, C, h, w); concatenate into one batch so
                # the model runs once per `batch_size` tiles instead of once per tile.
                inputs.append(
                    _make_model_input(
                        tile,
                        cfg.data.image_channels,
                        cfg.data.image_normalization,
                        device,
                        cfg.data.norm_mean,
                        cfg.data.norm_std,
                    )
                )

            probs = _infer_batch_prob(torch.cat(inputs, dim=0), model, cfg)

            for i, (y, x) in enumerate(chunk):
                valid_h = min(patch_h, H - y)
                valid_w = min(patch_w, W - x)
                win = window_t[:valid_h, :valid_w]
                prob[y:y+valid_h, x:x+valid_w] += probs[i, :valid_h, :valid_w] * win
                weight[y:y+valid_h, x:x+valid_w] += win

    return (prob / weight.clamp(min=1e-8)).cpu().numpy()


def predict_mask(
    img: np.ndarray,
    model: FiberSegmentationLitModule,
    cfg: AppConfig,
    device: torch.device,
) -> np.ndarray:
    """Run tiled inference on a single normalized grayscale image, returning a 0/255 uint8 mask.

    Thin wrapper over `predict_prob` that binarizes it per `cfg.inference.threshold_mode`:
    `"fixed"` (default) thresholds at `cfg.train.threshold`; `"hysteresis"` instead uses
    Canny-style two-threshold hysteresis (`cfg.inference.hysteresis_low`/`_high`) so thin
    low-confidence fibre continuations connected to a confident core survive instead of
    being severed by a single hard cutoff - see `tools.fiber_gap_repair.hysteresis_threshold_mask`;
    `"ridge"` first runs a Hessian ridge filter (`tools.fiber_gap_repair.enhance_ridges`)
    over the probability map and thresholds that response at `cfg.inference.ridge_threshold`,
    scoring how ridge-like a neighbourhood is rather than how bright a single pixel is.
    """
    return binarize_prob(predict_prob(img, model, cfg, device), cfg).astype(np.uint8) * 255


def binarize_prob(prob: np.ndarray, cfg: AppConfig) -> np.ndarray:
    """Boolean mask from a probability map per `cfg.inference.threshold_mode`
    (see `predict_mask`)."""
    mode = cfg.inference.threshold_mode
    if mode == "hysteresis":
        # Local import: fiber_gap_repair imports save_mask from this module, so importing
        # it at module load time would create a circular import.
        from .tools.fiber_gap_repair import hysteresis_threshold_mask

        mask = hysteresis_threshold_mask(
            prob, cfg.inference.hysteresis_low, cfg.inference.hysteresis_high
        )
    elif mode == "ridge":
        from .tools.fiber_gap_repair import enhance_ridges

        ridge = enhance_ridges(
            prob,
            method=cfg.inference.ridge_method,
            sigmas=tuple(cfg.inference.ridge_sigmas),
            device=cfg.inference.ridge_device,
        )
        mask = ridge > cfg.inference.ridge_threshold
    elif mode == "fixed":
        mask = prob > cfg.train.threshold
    else:
        raise ValueError(
            f"Unsupported inference.threshold_mode={mode!r}. "
            "Use 'fixed', 'hysteresis' or 'ridge'."
        )
    return np.asarray(mask, dtype=bool)


def load_predictor(checkpoint: str, cfg: AppConfig) -> tuple[FiberSegmentationLitModule, torch.device]:
    """Load a checkpoint in eval mode onto CUDA if available, else CPU.

    If the checkpoint carries validation-tuned thresholds (written after training by
    `tools.auto_tune`), they are applied to `cfg` in place, overriding the config's
    values - see `tuned_thresholds.apply_tuned`.
    """
    model = FiberSegmentationLitModule.load_from_checkpoint(
        checkpoint,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
    )
    applied = apply_tuned(cfg, model.tuned_thresholds)
    if applied:
        print("Using validation-tuned thresholds from checkpoint: " + "; ".join(applied))
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