# foreground_filter_sweep.py
"""Diagnostic tool for tuning data.patch_size / data.stride / data.min_foreground_fraction.

For every (patch_size, min_foreground_fraction) combination given on the command line,
this tiles every image/mask pair the same way TiledSegmentationDataset does, reports what
percentage of patches would be filtered out, and saves example patch/mask crops whose
foreground fraction sits just below (would be filtered) and just above (would be kept)
the min_foreground_fraction cutoff, so the cutoff can be judged visually.

The combinations to test can be listed either as CLI flags:
    python -m fiberseg.tools.foreground_filter_sweep --config configs/example.yaml \
        --patch-sizes 256 512 1024 \
        --foreground-fractions 0.0 0.001 0.01 0.05 \
        --out foreground_filter_sweep

...or under a `foreground_filter_sweep:` section of the same config YAML (see
configs/foreground_filter_sweep_example.yaml). CLI flags, if given, override the
config section.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..config import load_config
from ..dataset import (
    TiledSegmentationDataset,
    _cached_normalized_image,
    _cached_read,
    _hw,
    find_pairs,
)

# Mirrors TiledSegmentationDataset._make_tiles's per-split rng seeding
# (fiberseg/dataset.py). Keep in sync if that mapping changes.
_SPLIT_OFFSET = {"train": 0, "val": 1000, "test": 2000}


def _tile_positions(size: int, patch: int, stride: int) -> list[int]:
    """Reuses TiledSegmentationDataset._positions (a pure function; no `self` state
    is read in its body) so the tiling grid can never drift from what training uses."""
    return TiledSegmentationDataset._positions(None, size, patch, stride)


def to_uint8_display(img: np.ndarray) -> np.ndarray:
    """Convert an already-normalized [0, 1] float image to uint8 for saving/display."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def safe_name(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text)


def analyze_combo(
    pairs,
    patch_h: int,
    patch_w: int,
    stride_h: int,
    stride_w: int,
    min_foreground_fraction: float,
    keep_empty_probability: float,
    seed: int,
    split_offset: int,
) -> list[dict]:
    """Tile every pair and record each patch's foreground fraction and keep/filter
    decision, exactly mirroring TiledSegmentationDataset._make_tiles's filtering rule.
    """
    rng = random.Random(seed + split_offset)
    records: list[dict] = []

    for pair in pairs:
        img = _cached_read(str(pair.image_path))
        mask = _cached_read(str(pair.mask_path))
        H, W = img.shape[:2]

        for y in _tile_positions(H, patch_h, stride_h):
            for x in _tile_positions(W, patch_w, stride_w):
                m = mask[y:y + patch_h, x:x + patch_w]
                fg = float((m > 0).mean()) if m.size else 0.0

                kept = True
                if fg < min_foreground_fraction and rng.random() > keep_empty_probability:
                    kept = False

                records.append({
                    "pair": pair,
                    "y": y,
                    "x": x,
                    "h": patch_h,
                    "w": patch_w,
                    "foreground_fraction": fg,
                    "kept": kept,
                })

    return records


def save_example(
    group_dir: Path,
    idx: int,
    record: dict,
    min_foreground_fraction: float,
    dpi: int,
) -> None:
    pair = record["pair"]
    y, x, h, w = record["y"], record["x"], record["h"], record["w"]

    image_norm = _cached_normalized_image(str(pair.image_path))
    mask = _cached_read(str(pair.mask_path))

    img_crop = image_norm[y:y + h, x:x + w]
    mask_crop = (mask[y:y + h, x:x + w] > 0)

    display_img = to_uint8_display(img_crop)
    overlay = np.stack([display_img] * 3, axis=-1)
    overlay[mask_crop] = [255, 60, 60]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(display_img, cmap="gray", interpolation="nearest")
    axes[0].set_title("Image crop")
    axes[0].axis("off")

    axes[1].imshow(overlay, interpolation="nearest")
    axes[1].set_title(
        f"fg={record['foreground_fraction']:.4f}  cutoff={min_foreground_fraction:.4f}\n"
        f"{'KEPT' if record['kept'] else 'FILTERED'}"
    )
    axes[1].axis("off")

    fig.suptitle(f"{pair.image_path.stem}  y={y} x={x}")
    plt.tight_layout()

    fname = (
        f"{idx:02d}__{safe_name(pair.image_path.stem)}"
        f"__y{y}_x{x}__fg{record['foreground_fraction']:.4f}.png"
    )
    plt.savefig(group_dir / fname, dpi=dpi)
    plt.close(fig)


def save_boundary_examples(
    combo_dir: Path,
    records: list[dict],
    min_foreground_fraction: float,
    boundary_margin: float,
    n_examples: int,
    dpi: int,
) -> None:
    lo = max(0.0, min_foreground_fraction - boundary_margin)
    hi = min_foreground_fraction + boundary_margin

    just_under = [
        r for r in records
        if lo <= r["foreground_fraction"] < min_foreground_fraction
    ]
    just_over = [
        r for r in records
        if min_foreground_fraction <= r["foreground_fraction"] <= hi
    ]

    just_under.sort(key=lambda r: min_foreground_fraction - r["foreground_fraction"])
    just_over.sort(key=lambda r: r["foreground_fraction"] - min_foreground_fraction)

    for group_name, group in (
        ("just_under_cutoff", just_under),
        ("just_over_cutoff", just_over),
    ):
        group_dir = combo_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for idx, record in enumerate(group[:n_examples]):
            save_example(group_dir, idx, record, min_foreground_fraction, dpi)


def load_sweep_section(config_path: str) -> dict:
    """Reads the optional `foreground_filter_sweep:` section straight out of the
    config YAML. This section is specific to this diagnostic tool and is not part
    of AppConfig, so it's parsed independently instead of going through load_config.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("foreground_filter_sweep", {}) or {}


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "patch_size", "stride", "min_foreground_fraction",
        "total_tiles", "filtered_tiles", "filtered_fraction",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep patch_size / min_foreground_fraction combinations, report the "
            "percentage of patches each combination filters, and save example crops "
            "just below and just above the cutoff."
        )
    )
    parser.add_argument("--config", required=True, help="Config YAML (data section for images_dir/masks_dir/etc, plus an optional foreground_filter_sweep: section - see configs/foreground_filter_sweep_example.yaml).")
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=None, help="Square patch sizes to test, e.g. 256 512 1024. Overrides the config's foreground_filter_sweep.patch_sizes.")
    parser.add_argument("--strides", type=int, nargs="+", default=None, help="Stride per patch size (same order/length as --patch-sizes). Defaults to stride == patch_size (non-overlapping).")
    parser.add_argument("--foreground-fractions", type=float, nargs="+", default=None, help="min_foreground_fraction values to test, e.g. 0.0 0.001 0.01 0.05. Overrides the config's foreground_filter_sweep.foreground_fractions.")
    parser.add_argument("--keep-empty-probability", type=float, default=None, help="Same meaning as data.keep_empty_probability. Defaults to 0.0 (deterministic filtering) if not set here or in the config.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default=None, help="Which image/mask pairs to tile. Filtering only ever applies to 'train' during real training, so that's the default.")
    parser.add_argument("--boundary-margin", type=float, default=None, help="Absolute foreground-fraction band around the cutoff used to pick 'just under'/'just over' examples. Default 0.02.")
    parser.add_argument("--n-examples", type=int, default=None, help="Max number of example crops to save per side (under/over) per combination. Default 8.")
    parser.add_argument("--out", default=None, help="Output directory. Default 'foreground_filter_sweep'.")
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args()

    sweep_cfg = load_sweep_section(args.config)

    def resolved(cli_value, key, default=None):
        return cli_value if cli_value is not None else sweep_cfg.get(key, default)

    patch_sizes = resolved(args.patch_sizes, "patch_sizes")
    strides = resolved(args.strides, "strides")
    foreground_fractions = resolved(args.foreground_fractions, "foreground_fractions")
    keep_empty_probability = resolved(args.keep_empty_probability, "keep_empty_probability", 0.0)
    split = resolved(args.split, "split", "train")
    boundary_margin = resolved(args.boundary_margin, "boundary_margin", 0.02)
    n_examples = resolved(args.n_examples, "n_examples", 8)
    out = resolved(args.out, "out", "foreground_filter_sweep")
    dpi = resolved(args.dpi, "dpi", 150)

    if not patch_sizes:
        raise ValueError("patch sizes must be given via --patch-sizes or the config's foreground_filter_sweep.patch_sizes.")
    if not foreground_fractions:
        raise ValueError("foreground fractions must be given via --foreground-fractions or the config's foreground_filter_sweep.foreground_fractions.")
    if strides is not None and len(strides) != len(patch_sizes):
        raise ValueError("strides must have the same number of values as patch_sizes.")

    cfg = load_config(args.config).data

    all_pairs = find_pairs(cfg)
    pairs = all_pairs if split == "all" else [p for p in all_pairs if p.split == split]
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for split={split!r}.")

    split_offset = _SPLIT_OFFSET.get(split, 3000)

    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)

    strides = strides if strides is not None else list(patch_sizes)

    summary_rows: list[dict] = []

    for patch, stride in zip(patch_sizes, strides):
        patch_h, patch_w = _hw(patch)
        stride_h, stride_w = _hw(stride)

        for min_fg in foreground_fractions:
            records = analyze_combo(
                pairs, patch_h, patch_w, stride_h, stride_w,
                min_fg, keep_empty_probability, cfg.seed, split_offset,
            )

            total = len(records)
            filtered = sum(1 for r in records if not r["kept"])
            filtered_fraction = filtered / total if total else 0.0

            summary_rows.append({
                "patch_size": patch,
                "stride": stride,
                "min_foreground_fraction": min_fg,
                "total_tiles": total,
                "filtered_tiles": filtered,
                "filtered_fraction": filtered_fraction,
            })

            print(
                f"patch={patch} stride={stride} min_fg={min_fg:.4f} -> "
                f"{filtered}/{total} filtered ({filtered_fraction:.1%})"
            )

            combo_dir = out_root / f"patch{patch}_stride{stride}_fg{min_fg:g}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            save_boundary_examples(
                combo_dir, records, min_fg, boundary_margin, n_examples, dpi,
            )

    summary_path = out_root / "summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
