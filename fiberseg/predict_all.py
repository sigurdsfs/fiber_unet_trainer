# predict_all.py
"""Batch tiled inference over every input image referenced by a config.

Loads a checkpoint once via `predict_tiles.load_predictor`, then runs
`predict_tiles.predict_mask` for each image found under `data.images_dir`/`data.image_glob`
(same discovery/exclusion rule as `dataset.find_pairs`, but no matching mask is required
since this is inference, not training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dataset import IMG_EXTENSIONS, _normalize_image, _read_gray
from .predict_tiles import load_predictor, predict_mask, save_mask


def find_images(images_dir: Path, image_glob: str) -> list[Path]:
    """List input images under `images_dir` matching `image_glob`, excluding `*_mask` files."""
    return sorted(
        p for p in images_dir.glob(image_glob)
        if p.suffix.lower() in IMG_EXTENSIONS
        and not p.stem.endswith("_mask")
    )


def main():
    """CLI entry point: predict a mask for every image in the config and write it to `--out-dir`."""
    parser = argparse.ArgumentParser(
        description="Run tiled prediction on every input image referenced by a config's "
        "data.images_dir/data.image_glob."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--suffix",
        default="_pred.tif",
        help="Filename suffix appended to each image stem for the output mask (default: _pred.tif).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    images_dir = Path(cfg.data.images_dir)
    images = find_images(images_dir, cfg.data.image_glob)

    if not images:
        raise FileNotFoundError(
            f"No input images found in {images_dir} using glob {cfg.data.image_glob!r}."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] Predicting {image_path.name} ...")

        img = _normalize_image(_read_gray(image_path))
        mask = predict_mask(img, model, cfg, device)

        save_mask(mask, out_dir / f"{image_path.stem}{args.suffix}")

    print(f"Done. Wrote {len(images)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
