# list_split_files.py
"""Report which image filenames were assigned to each train/val/test split for a config."""
from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..dataset import split_filenames


def main():
    parser = argparse.ArgumentParser(
        description="List the image filenames assigned to each train/val/test split for a config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", help="Optional path to write the result as JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = split_filenames(cfg.data)

    for split, names in result.items():
        print(f"{split}: {len(names)} files")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote split file lists to {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
