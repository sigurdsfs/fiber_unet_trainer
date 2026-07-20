# preview_augmentations.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..augmentations import _build_transform_list
from ..config import load_config
from ..dataset import _normalize_image, _read_gray, find_pairs


def to_uint8_display(img: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] image to uint8 [0, 255] for saving/display only."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def build_replay_transform(items: list[dict[str, Any]] | None):
    """Build Albumentations ReplayCompose so we can see what was applied.

    Reuses the same validated transform construction as the training pipeline
    (fiberseg.augmentations.build_transform), so this preview can never
    silently drift from what training actually does - e.g. a GaussNoise entry
    missing 'std_range' is rejected here exactly like it is during training.
    """
    transforms = _build_transform_list(items)
    if not transforms:
        return None
    return A.ReplayCompose(transforms)


def get_applied_transform_names(replay: dict[str, Any]) -> list[str]:
    """Extract names of transforms that were actually applied."""
    applied = []

    for transform in replay.get("transforms", []):
        if transform.get("applied", False):
            name = transform.get("__class_fullname__", "Unknown")
            name = name.split(".")[-1]
            applied.append(name)

    return applied


def safe_name(text: str, max_len: int = 120) -> str:
    """Make a safe filename component."""
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            keep.append(ch)
        else:
            keep.append("_")

    out = "".join(keep).strip("_")
    return out[:max_len] if out else "none"


def random_crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
):
    """Take a random crop from image and mask at the same location."""
    h, w = image.shape[:2]

    if crop_size <= 0:
        return image, mask

    if h <= crop_size or w <= crop_size:
        return image, mask

    y = int(rng.integers(0, h - crop_size + 1))
    x = int(rng.integers(0, w - crop_size + 1))

    return (
        image[y : y + crop_size, x : x + crop_size],
        mask[y : y + crop_size, x : x + crop_size],
    )


def save_raw_crop_outputs(
    out_dir: Path,
    img_stem: str,
    aug_idx: int,
    applied_file_text: str,
    crop_image: np.ndarray,
    crop_mask: np.ndarray,
    aug_image: np.ndarray,
    aug_mask: np.ndarray,
):
    """Save actual pixel-resolution crops, not downsampled matplotlib figures."""
    prefix = f"{img_stem}__aug{aug_idx:02d}__{applied_file_text}"

    Image.fromarray(crop_image).save(out_dir / f"{prefix}__original_crop.png")
    Image.fromarray(crop_mask).save(out_dir / f"{prefix}__original_mask_crop.png")
    Image.fromarray(aug_image).save(out_dir / f"{prefix}__augmented_crop.png")
    Image.fromarray(aug_mask).save(out_dir / f"{prefix}__augmented_mask_crop.png")


def main():
    parser = argparse.ArgumentParser(
        description="Preview training augmentations on image/mask crops."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="augmentation_preview")
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--n-aug", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--save-raw-crops",
        action="store_true",
        help="Save original and augmented crops as separate full-resolution PNG files.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.data
    train_augs = cfg.augmentations.get("train", [])
    aug = build_replay_transform(train_augs)

    rng = np.random.default_rng(data_cfg.seed)

    # Only preview pairs from the train split: val/test never get augmented
    # (see the config's augmentations.val / augmentations.test), so previewing
    # them here would be misleading.
    pairs = [p for p in find_pairs(data_cfg) if p.split == "train"][: args.n_images]

    if not pairs:
        raise RuntimeError(
            "No training image/mask pairs found. Check images_dir, masks_dir, "
            "image_glob, and mask_pattern in the config."
        )

    for pair in pairs:
        # Same preprocessing as training: normalize the whole image once (same
        # percentiles as TiledSegmentationDataset / predict_tiles.py), then crop.
        # Normalizing per-crop instead would give every preview its own contrast
        # stretch and misrepresent what the model actually trains on.
        image = _normalize_image(_read_gray(pair.image_path))
        mask = (_read_gray(pair.mask_path) > 0).astype(np.float32)

        for aug_idx in range(args.n_aug):
            crop_image, crop_mask = random_crop_pair(
                image=image,
                mask=mask,
                crop_size=args.crop_size,
                rng=rng,
            )

            if aug is None:
                aug_image = crop_image
                aug_mask = crop_mask
                applied_names = []
            else:
                transformed = aug(image=crop_image, mask=crop_mask)
                aug_image = transformed["image"]
                aug_mask = transformed["mask"]
                applied_names = get_applied_transform_names(transformed["replay"])

            # Same final step as TiledSegmentationDataset.__getitem__: clip
            # augmentation output back into [0, 1] before anything downstream.
            aug_image = np.clip(aug_image, 0.0, 1.0).astype(np.float32)

            if applied_names:
                # Wrap onto multiple lines so long combinations don't overlap
                # the neighboring subplot title.
                applied_text = " +\n".join(
                    " + ".join(applied_names[i:i + 2])
                    for i in range(0, len(applied_names), 2)
                )
                applied_file_text = safe_name("_".join(applied_names))
            else:
                applied_text = "None"
                applied_file_text = "none"

            display_crop_image = to_uint8_display(crop_image)
            display_crop_mask = (crop_mask * 255).astype(np.uint8)
            display_aug_image = to_uint8_display(aug_image)
            display_aug_mask = (aug_mask * 255).astype(np.uint8)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(display_crop_image, cmap="gray", interpolation="nearest")
            axes[0, 0].set_title(f"Original crop\n{pair.image_path.name}")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(display_crop_mask, cmap="gray", interpolation="nearest")
            axes[0, 1].set_title("Original mask crop")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(display_aug_image, cmap="gray", interpolation="nearest")
            axes[1, 0].set_title(f"Augmented crop\n{applied_text}", fontsize=9)
            axes[1, 0].axis("off")

            axes[1, 1].imshow(display_aug_mask, cmap="gray", interpolation="nearest")
            axes[1, 1].set_title(f"Augmented mask crop\n{applied_text}", fontsize=9)
            axes[1, 1].axis("off")

            plt.tight_layout()

            out_name = (
                f"{pair.image_path.stem}"
                f"__aug{aug_idx:02d}"
                f"__{applied_file_text}"
                f"__crop{args.crop_size}.png"
            )

            out_path = out_dir / out_name
            plt.savefig(out_path, dpi=args.dpi)
            plt.close(fig)

            if args.save_raw_crops:
                save_raw_crop_outputs(
                    out_dir=out_dir,
                    img_stem=pair.image_path.stem,
                    aug_idx=aug_idx,
                    applied_file_text=applied_file_text,
                    crop_image=display_crop_image,
                    crop_mask=display_crop_mask,
                    aug_image=display_aug_image,
                    aug_mask=display_aug_mask,
                )

            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
