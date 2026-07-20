# compute_dataset_stats.py
"""Compute per-channel mean/std over the TRAINING split for image_normalization: "dataset".

For from-scratch training (`encoder_weights: null`) there is no pretrained input
distribution to match, so standardizing with the dataset's own statistics is the
natural choice. This computes those statistics over the training split only (never
val/test, to avoid leaking evaluation data), in the same percentile-normalized [0,1]
space the model actually consumes.

Run:
    python -m fiberseg.tools.compute_dataset_stats --config <cfg>            # print only
    python -m fiberseg.tools.compute_dataset_stats --config <cfg> --write    # write into config

--write performs a targeted edit of the `data:` block (setting image_normalization,
norm_mean, norm_std) while leaving the rest of the file - including comments - intact.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..config import load_config
from ..dataset import compute_normalization_stats


def _fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def _data_block_indent(lines: list[str], data_idx: int) -> str:
    """Indentation of keys directly under the top-level `data:` mapping."""
    for line in lines[data_idx + 1:]:
        if line.strip() and not line.lstrip().startswith("#"):
            return line[: len(line) - len(line.lstrip())]
    return "  "


def _write_stats_into_config(path: Path, mean: list[float], std: list[float]) -> None:
    """Set image_normalization/norm_mean/norm_std under the top-level `data:` key.

    Replaces those keys if present, otherwise inserts them at the top of the data
    block. Only the `data:` section is touched; comments elsewhere are preserved.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    data_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"^data:\s*(#.*)?$", ln.rstrip("\n"))),
        None,
    )
    if data_idx is None:
        raise SystemExit(
            f"Could not find a top-level 'data:' section in {path}; not writing. "
            "Paste the printed values manually."
        )

    indent = _data_block_indent(lines, data_idx)
    # Extent of the data block: until the next top-level (unindented) key.
    end_idx = len(lines)
    for i in range(data_idx + 1, len(lines)):
        stripped = lines[i].rstrip("\n")
        if stripped and not stripped[0].isspace() and not stripped.lstrip().startswith("#"):
            end_idx = i
            break

    new_values = {
        "image_normalization": '"dataset"',
        "norm_mean": _fmt_list(mean),
        "norm_std": _fmt_list(std),
    }

    # Replace existing keys in place.
    remaining = dict(new_values)
    for i in range(data_idx + 1, end_idx):
        m = re.match(rf"^{re.escape(indent)}([A-Za-z_][\w]*):", lines[i])
        if m and m.group(1) in remaining:
            key = m.group(1)
            lines[i] = f"{indent}{key}: {remaining.pop(key)}\n"

    # Insert any keys that weren't already present, right after `data:`.
    if remaining:
        insertion = "".join(f"{indent}{k}: {v}\n" for k, v in remaining.items())
        lines.insert(data_idx + 1, insertion)

    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compute training-split mean/std for image_normalization: 'dataset'."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the computed stats (and image_normalization: dataset) into the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.model.encoder_weights is not None:
        print(
            f"Note: encoder_weights={cfg.model.encoder_weights!r} is set. Dataset-specific "
            "stats are intended for from-scratch training; a pretrained encoder usually "
            "wants image_normalization: 'imagenet' instead."
        )

    print(f"Computing mean/std over the training split ({cfg.data.images_dir}) ...")
    mean, std = compute_normalization_stats(cfg.data)

    print("=" * 60)
    print(f"  norm_mean: {_fmt_list(mean)}")
    print(f"  norm_std:  {_fmt_list(std)}")
    print("=" * 60)

    if args.write:
        _write_stats_into_config(Path(args.config), mean, std)
        print(f"Wrote image_normalization: dataset + norm_mean/norm_std into {args.config}")
    else:
        print("Add these under `data:` and set image_normalization: \"dataset\" "
              "(or re-run with --write).")


if __name__ == "__main__":
    main()
