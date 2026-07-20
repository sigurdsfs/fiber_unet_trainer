from __future__ import annotations

from typing import Any

import albumentations as A


def _build_transform_list(items: list[dict[str, Any]] | None) -> list[Any]:
    """Validate and instantiate the Albumentations transforms in a config list.

    Shared by build_transform() and the preview_augmentations tool so both
    enforce the same validation and can never silently drift apart.
    """
    if not items:
        return []
    transforms = []
    for item in items:
        item = dict(item)
        name = item.pop("name")
        if not hasattr(A, name):
            raise ValueError(f"Unknown Albumentations transform: {name}")
        if name == "GaussNoise" and "std_range" not in item:
            raise ValueError(
                "GaussNoise requires an explicit 'std_range'. This pipeline normalizes "
                "images to [0, 1] before augmentation, and Albumentations' default "
                "std_range=(0.2, 0.44) is calibrated for a much wider intensity range - "
                "applied here it saturates the image with noise. Set e.g. "
                "'std_range: [0.02, 0.08]' for a sane noise level at [0, 1] scale."
            )
        cls = getattr(A, name)
        transforms.append(cls(**item))
    return transforms


def build_transform(items: list[dict[str, Any]] | None):
    """Build an Albumentations Compose from a list of config dicts.

    Example item:
        {"name": "HorizontalFlip", "p": 0.5}

    Albumentations applies geometric transforms consistently to image and mask,
    while image-only transforms affect only the image.
    """
    transforms = _build_transform_list(items)
    if not transforms:
        return None
    return A.Compose(transforms)
