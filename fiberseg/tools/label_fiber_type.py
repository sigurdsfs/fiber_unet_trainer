# fiberseg/tools/label_fiber_type.py
"""Interactively review image/mask pairs and label each by fiber type.

Same interaction model as `label_pairs.py`, but the two labels are fiber types
instead of quality: Chrysotile or Amphibole. Opens one image/mask/overlay
triplet at a time in a native, resizable window (the TkAgg backend, not a
notebook widget), so the window's own toolbar gives you real zoom/pan/reset -
drag to zoom, scroll or the magnifier tool to zoom in/out.

Click the figure once so it has keyboard focus, then press:

    c - label the pair "chrysotile", advance to the next unlabeled pair
    a - label the pair "amphibole", advance to the next unlabeled pair
    u - undo the last label and jump back to it
    r - reset zoom/pan back to the original view (matplotlib's default Home key)
    q - quit

Labels are saved incrementally to a CSV, so you can stop and resume later -
already-labeled pairs are skipped automatically unless --relabel-all is given.

Run with:
    python -m fiberseg.tools.label_fiber_type --config configs/simple_sweep.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_config
from .label_pairs import PairLabeler, find_pairs_and_missing

KEY_TO_LABEL = {"c": "chrysotile", "a": "amphibole"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively label image/mask pairs by fiber type (Chrysotile/Amphibole) in a zoomable window."
    )
    parser.add_argument("--config", required=True, help="Project YAML config (reads its data: section).")
    parser.add_argument("--images-dir", default=None, help="Override data.images_dir.")
    parser.add_argument("--masks-dir", default=None, help="Override data.masks_dir.")
    parser.add_argument("--image-glob", default=None, help="Override data.image_glob.")
    parser.add_argument("--mask-pattern", default=None, help="Override data.mask_pattern.")
    parser.add_argument(
        "--labels-csv",
        default="notebooks/fiber_type_labels.csv",
        help="Where to save/resume labels.",
    )
    parser.add_argument("--relabel-all", action="store_true", help="Ignore existing labels and start over.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_dir = Path(args.images_dir or cfg.data.images_dir)
    masks_dir = Path(args.masks_dir or cfg.data.masks_dir)
    image_glob = args.image_glob or cfg.data.image_glob
    mask_pattern = args.mask_pattern or cfg.data.mask_pattern

    print(f"images_dir:   {images_dir.resolve()}")
    print(f"masks_dir:    {masks_dir.resolve()}")
    print(f"image_glob:   {image_glob!r}")
    print(f"mask_pattern: {mask_pattern!r}")

    pairs, unmatched = find_pairs_and_missing(images_dir, masks_dir, image_glob, mask_pattern)
    print(f"{len(pairs)} image/mask pairs found")
    print(f"{len(unmatched)} images have no matching mask")
    for img in unmatched:
        print(f"  missing mask for: {img.name}")

    if not pairs:
        print("No pairs to label.")
        return

    labels_csv = Path(args.labels_csv)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)

    labeler = PairLabeler(pairs, labels_csv, args.relabel_all, key_to_label=KEY_TO_LABEL)
    labeler.run()

    df = pd.read_csv(labels_csv) if labels_csv.exists() else pd.DataFrame(columns=["image", "label"])
    print("\nFinal label counts:")
    print(df["label"].value_counts().to_string() if len(df) else "(none labeled)")
    print(f"\nLabels saved to: {labels_csv.resolve()}")


if __name__ == "__main__":
    main()
