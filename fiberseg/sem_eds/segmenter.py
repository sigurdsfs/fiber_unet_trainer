"""Segmentation backends: image (H, W) float [0, 1] -> instance label image (H, W) int32.

FiberSegSegmenter replaces the handoff's LightningSegmenter. It goes through
fiberseg.predict_tiles (load_predictor / predict_mask), so preprocessing is IDENTICAL to
training: percentile normalisation (_normalize_image), image_channels, image_normalization
(+ dataset stats), Gaussian tile blending, reflect padding, optional TTA and the binarization
mode (inference.threshold_mode) all come from the checkpoint's training config. The model
outputs one-channel raw logits (see models.py), so there is no class-index choice to make.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, transform

from .settings import SemEdsConfig


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected components smaller than min_px (version-independent)."""
    lab, _ = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def mask_to_instances(fg: np.ndarray, min_px: int) -> np.ndarray:
    # TODO: touching/crossing fibres merge into one component here. If that matters for
    # counting, split with a skeleton-based or watershed step before measuring.
    return measure.label(remove_small(fg, min_px), connectivity=2).astype(np.int32)


class FiberSegSegmenter:
    def __init__(self, cfg: SemEdsConfig):
        from ..config import load_config
        from ..predict_tiles import load_predictor

        if not cfg.ckpt:
            raise ValueError("settings.ckpt is empty")
        self.cfg = cfg
        self.app_cfg = load_config(cfg.seg_config)
        self.model, self.device = load_predictor(cfg.ckpt, self.app_cfg)

    def foreground(self, img: np.ndarray) -> np.ndarray:
        """Boolean fibre mask at the acquisition resolution. Binarization goes through
        predict_mask, so inference.threshold_mode (fixed / hysteresis / ridge) is honoured."""
        from ..dataset import _normalize_image
        from ..predict_tiles import predict_mask

        x = _normalize_image(img)
        scale = 1.0
        if self.cfg.training_pixel_size_um > 0:
            # >1 upsamples: acquisition pixels are coarser than the training pixels.
            scale = self.cfg.pixel_size_um / self.cfg.training_pixel_size_um
        if abs(scale - 1.0) > 0.02:
            x = transform.rescale(x, scale, order=1, anti_aliasing=scale < 1, preserve_range=True)
        fg = predict_mask(x.astype(np.float32), self.model, self.app_cfg, self.device) > 0
        if fg.shape != img.shape:
            fg = transform.resize(fg, img.shape, order=0, preserve_range=True, anti_aliasing=False)
        return fg.astype(bool)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return mask_to_instances(self.foreground(img), self.cfg.min_object_px)


class ThresholdSegmenter:
    """Stand-in model for --dry-run without a checkpoint."""

    def __init__(self, cfg: SemEdsConfig):
        self.cfg = cfg

    def __call__(self, img):
        return mask_to_instances(img > 0.5, self.cfg.min_object_px)
