# rank_uncertainty.py
"""Rank unlabeled images by model uncertainty to guide the next annotation round.

The single biggest lever on a 28-image dataset is more labeled data - but only if
the new images are informative. This runs a trained checkpoint over a folder of
UNLABELED images, scores each by how uncertain the model is (probabilities near
0.5 = the model is guessing), and writes a CSV sorted most-uncertain first.
Annotate the top of that list next: active learning typically reaches a target
accuracy with far fewer labels than random selection.

Run:
    python -m fiberseg.tools.rank_uncertainty --config <cfg> --checkpoint <best.ckpt> \
        --images-dir <folder-of-unlabeled-tifs> --out uncertainty_ranking.csv

Note: point --images-dir at images WITHOUT masks (a candidate pool), not your
current training set.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..config import load_config
from ..dataset import IMG_EXTENSIONS, _normalize_image, _read_gray
from ..predict_tiles import load_predictor, predict_prob


def _uncertainty_scores(prob: np.ndarray) -> dict[str, float]:
    """Per-image uncertainty summaries from a [0,1] probability map.

    - mean_entropy: mean binary entropy; high when many pixels sit near 0.5.
    - boundary_uncertainty: fraction of pixels in the ambiguous [0.3, 0.7] band,
      where the model is neither confidently fiber nor confidently background.
    - predicted_fiber_fraction: coverage sanity check (very high/low can flag
      out-of-distribution images worth a look).
    """
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    band = ((prob >= 0.3) & (prob <= 0.7)).mean()
    return {
        "mean_entropy": float(entropy.mean()),
        "boundary_uncertainty": float(band),
        "predicted_fiber_fraction": float((prob > 0.5).mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rank unlabeled images by model uncertainty for active-learning "
        "annotation prioritization."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Folder of UNLABELED candidate images (no masks needed).",
    )
    parser.add_argument("--glob", default="*.tif", help="Glob for candidate images.")
    parser.add_argument("--out", default="uncertainty_ranking.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_predictor(args.checkpoint, cfg)

    images_dir = Path(args.images_dir)
    images = sorted(
        p for p in images_dir.glob(args.glob)
        if p.suffix.lower() in IMG_EXTENSIONS and not p.stem.endswith("_mask")
    )
    if not images:
        raise SystemExit(f"No candidate images in {images_dir} matching {args.glob!r}.")

    rows = []
    for i, path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {path.name}")
        img = _normalize_image(_read_gray(path))
        prob = predict_prob(img, model, cfg, device)
        rows.append({"image": path.name, **_uncertainty_scores(prob)})

    # Most uncertain first: highest mean entropy is the strongest signal.
    rows.sort(key=lambda r: r["mean_entropy"], reverse=True)

    out_path = Path(args.out)
    fieldnames = ["rank", "image", "mean_entropy", "boundary_uncertainty",
                  "predicted_fiber_fraction"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    print("=" * 60)
    print(f"Wrote {len(rows)} ranked images to {out_path}")
    print("Annotate these first (most uncertain):")
    for row in rows[:10]:
        print(f"  {row['mean_entropy']:.4f}  {row['image']}")


if __name__ == "__main__":
    main()
