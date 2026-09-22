# error_overlays.py
"""Render TP/FP/FN overlays for the worst-scoring images of a predict_all.py run.

A handful of images usually carry a big share of the error (on the se_resnet152
baseline the worst 5 test images held ~34% of all wrong pixels), and the useful question
for each is *why*: label mistakes, artefacts/scratches the model fires on, a fiber type
or contrast it has never seen, or a pixel-size outlier. That needs eyes on the image.

Colors: green = TP (fiber found), red = FP (predicted fiber, not in GT),
blue = FN (GT fiber missed). Thin classes are max-pooled when downsizing for display so
1-px fibers stay visible.

Run:
    python -m fiberseg.tools.error_overlays --config <cfg> --pred-dir predictions/run \
        --split test --worst 10
Writes <pred-dir>/error_overlays/<rank>_<image>.png (or --out-dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import load_config
from ..dataset import SPLIT_DIRS, _normalize_image, _read_gray, find_pairs

_COLORS_BGR = {"tp": (60, 200, 60), "fp": (40, 40, 230), "fn": (230, 120, 30)}


def _shrink_max(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Downsize a boolean mask so any foreground in a block survives (max-pool-like)."""
    return cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0


def render_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, max_side: int = 2048,
                   caption: str = "") -> np.ndarray:
    """BGR uint8 overlay of TP/FP/FN on the (percentile-normalized) grayscale image."""
    h, w = gt.shape
    s = min(1.0, max_side / max(h, w))
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    gray = (np.clip(_normalize_image(img), 0, 1) * 255).astype(np.uint8)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    g, p = gt > 0, pred > 0
    # Draw FN, then FP, then TP, so correctly found fiber wins where downsized classes touch.
    for name, m in (("fn", g & ~p), ("fp", p & ~g), ("tp", g & p)):
        out[_shrink_max(m, size)] = _COLORS_BGR[name]
    if caption:
        bar = np.full((34, out.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        out = np.vstack([bar, out])
    return out


def main():
    parser = argparse.ArgumentParser(description="TP/FP/FN overlays for the worst images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir", required=True, help="predict_all.py --out-dir.")
    parser.add_argument("--metrics", default=None, help="Default: <pred-dir>/metrics.csv.")
    parser.add_argument("--split", default="test", help="train/val/test or 'all'.")
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--sort-by", default=None,
                        help="Metric column to rank by, ascending (default: raw_dice or dice).")
    parser.add_argument("--suffix", default="_pred.tif")
    parser.add_argument("--postprocessed", action="store_true",
                        help="Overlay the postprocessed/ masks instead of the raw ones.")
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(args.pred_dir)
    df = pd.read_csv(args.metrics or pred_dir / "metrics.csv")
    sort_by = args.sort_by or ("raw_dice" if "raw_dice" in df.columns else "dice")
    if args.split != "all":
        df = df[df["split"] == args.split]
    df = df.sort_values(sort_by).head(args.worst)

    pairs = {p.image_path.name: p for p in find_pairs(cfg.data)}
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir / "error_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pred_dir / "postprocessed" if args.postprocessed else pred_dir

    for rank, row in enumerate(df.itertuples(index=False), start=1):
        pair = pairs.get(row.image)
        if pair is None:
            print(f"skip {row.image}: not found via config's data.images_dir")
            continue
        # The split the prediction was WRITTEN under (metrics.csv), not today's find_pairs
        # split - they differ if images were added/removed since that run.
        pred_path = base / SPLIT_DIRS[row.split] / f"{pair.image_path.stem}{args.suffix}"
        if not pred_path.exists():
            print(f"skip {row.image}: no prediction at {pred_path}")
            continue
        img, gt, pred = (_read_gray(p) for p in (pair.image_path, pair.mask_path, pred_path))
        r = row._asdict()
        pre = "raw_" if "raw_dice" in r else ""
        caption = (f"#{rank} {row.image} | dice {r[pre + 'dice']:.3f}  "
                   f"P {r[pre + 'precision']:.3f}  R {r[pre + 'recall']:.3f}  "
                   "| green TP  red FP  blue FN")
        out = render_overlay(img, gt, pred, args.max_side, caption)
        dst = out_dir / f"{rank:02d}_{pair.image_path.stem}.png"
        cv2.imwrite(str(dst), out)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
