# fiberseg/tools/label_pairs.py
"""Interactively review image/mask pairs and label each as good/bad/redo.

Opens one image/mask/overlay triplet at a time in a native, resizable window (the
TkAgg backend, not a notebook widget), so the window's own toolbar gives you real
zoom/pan/reset - drag to zoom, scroll or the magnifier tool to zoom in/out.

Click the figure once so it has keyboard focus, then press:

    g - label the pair "good", advance to the next unlabeled pair
    b - label the pair "bad", advance to the next unlabeled pair
    e - label the pair "redo" (needs to be reworked/re-annotated), advance
    u - undo the last label and jump back to it
    r - reset zoom/pan back to the original view (matplotlib's default Home key)
    q - quit

Labels are saved incrementally to a CSV, so you can stop and resume later -
already-labeled pairs are skipped automatically unless --relabel-all is given.

Run with:
    python -m fiberseg.tools.label_pairs --config configs/simple_sweep.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from ..config import load_config

IMG_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
KEY_TO_LABEL = {"g": "good", "b": "bad", "e": "redo"}


def read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path).convert("L"))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.size == 0:
        return img
    lo, hi = np.percentile(img, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def find_pairs_and_missing(
    images_dir: Path, masks_dir: Path, image_glob: str, mask_pattern: str
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Mirrors fiberseg.dataset.find_pairs's pairing logic (glob + `_mask` exclusion +
    mask_pattern lookup), but also returns images with no matching mask instead of
    just warning about them.
    """
    image_paths = sorted(
        p for p in images_dir.glob(image_glob)
        if p.suffix.lower() in IMG_EXTENSIONS and not p.stem.endswith("_mask")
    )

    pairs = []
    unmatched = []

    for img in image_paths:
        mask_name = mask_pattern.format(stem=img.stem, suffix=img.suffix, name=img.name)
        mask = masks_dir / mask_name
        if mask.exists():
            pairs.append((img, mask))
        else:
            unmatched.append(img)

    return pairs, unmatched


class PairLabeler:
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        labels_csv: Path,
        relabel_all: bool,
        key_to_label: dict[str, str] | None = None,
    ):
        self.pairs = pairs
        self.labels_csv = labels_csv
        self.key_to_label = key_to_label or KEY_TO_LABEL
        self.order = [img_path.name for img_path, _ in pairs]
        self.pair_by_name = {img_path.name: (img_path, mask_path) for img_path, mask_path in pairs}
        self.history: list[str] = []

        if labels_csv.exists() and not relabel_all:
            df = pd.read_csv(labels_csv)
            self.labels: dict[str, str] = dict(zip(df["image"], df["label"]))
        else:
            self.labels = {}

        self.idx = self._next_unlabeled(0)

        # Free up 'g', which matplotlib binds to grid-toggle by default, so it acts
        # as a pure label key. 'r' is left bound to matplotlib's default Home
        # action (reset zoom/pan to the original view).
        plt.rcParams["keymap.grid"] = []
        plt.rcParams["keymap.grid_minor"] = []

        self.keymap_help = "  ".join(f"{k}={v}" for k, v in self.key_to_label.items())

        self.fig, self.axes = plt.subplots(1, 3, figsize=(22, 9))
        self.fig.canvas.manager.set_window_title(
            f"Label image/mask pairs -- {self.keymap_help}  u=undo  r=reset zoom  q=quit"
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _next_unlabeled(self, start: int) -> int:
        i = start
        while i < len(self.order) and self.order[i] in self.labels:
            i += 1
        return i

    def _save(self) -> None:
        pd.DataFrame(
            {"image": list(self.labels.keys()), "label": list(self.labels.values())}
        ).to_csv(self.labels_csv, index=False)

    def _label_current(self, label: str) -> None:
        if self.idx >= len(self.order):
            return
        name = self.order[self.idx]
        self.labels[name] = label
        self.history.append(name)
        self._save()
        self.idx = self._next_unlabeled(self.idx + 1)
        self.render()

    def _undo(self) -> None:
        if not self.history:
            return
        name = self.history.pop()
        self.labels.pop(name, None)
        self._save()
        self.idx = self.order.index(name)
        self.render()

    def _on_key(self, event) -> None:
        if event.key in self.key_to_label:
            self._label_current(self.key_to_label[event.key])
        elif event.key == "u":
            self._undo()
        elif event.key == "q":
            plt.close(self.fig)

    def _counts(self) -> dict[str, int]:
        counts = {v: 0 for v in self.key_to_label.values()}
        for v in self.labels.values():
            counts[v] = counts.get(v, 0) + 1
        return counts

    def render(self) -> None:
        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        counts = self._counts()
        counts_str = "  ".join(f"{label}={counts[label]}" for label in self.key_to_label.values())

        if self.idx >= len(self.order):
            self.fig.suptitle(f"All pairs labeled -- {counts_str} out of {len(self.order)}.")
            self.fig.canvas.draw_idle()
            return

        name = self.order[self.idx]
        img_path, mask_path = self.pair_by_name[name]
        image = normalize_for_display(read_gray(img_path))
        mask = read_gray(mask_path) > 0

        # interpolation="nearest": show true pixels when zoomed in, instead of
        # matplotlib's default blurring/smoothing between them.
        self.axes[0].imshow(image, cmap="gray", interpolation="nearest")
        self.axes[0].set_title(img_path.name)
        self.axes[1].imshow(mask, cmap="gray", interpolation="nearest")
        self.axes[1].set_title(mask_path.name)
        self.axes[2].imshow(image, cmap="gray", interpolation="nearest")
        self.axes[2].imshow(
            np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.5, interpolation="nearest"
        )
        self.axes[2].set_title("overlay")
        for ax in self.axes:
            ax.axis("off")

        already = f"  (already labeled: {self.labels[name]})" if name in self.labels else ""
        self.fig.suptitle(
            f"{self.idx + 1}/{len(self.order)}  {name}{already}   |   "
            f"{self.keymap_help}  u=undo  r=reset zoom  q=quit   |   "
            f"{counts_str}"
        )
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        self.render()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively label image/mask pairs as good/bad/redo in a zoomable window."
    )
    parser.add_argument("--config", required=True, help="Project YAML config (reads its data: section).")
    parser.add_argument("--images-dir", default=None, help="Override data.images_dir.")
    parser.add_argument("--masks-dir", default=None, help="Override data.masks_dir.")
    parser.add_argument("--image-glob", default=None, help="Override data.image_glob.")
    parser.add_argument("--mask-pattern", default=None, help="Override data.mask_pattern.")
    parser.add_argument(
        "--labels-csv",
        default="notebooks/pair_labels.csv",
        help="Where to save/resume labels (default continues the existing notebook's labels).",
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

    labeler = PairLabeler(pairs, labels_csv, args.relabel_all)
    labeler.run()

    df = pd.read_csv(labels_csv) if labels_csv.exists() else pd.DataFrame(columns=["image", "label"])
    print("\nFinal label counts:")
    print(df["label"].value_counts().to_string() if len(df) else "(none labeled)")
    print(f"\nLabels saved to: {labels_csv.resolve()}")


if __name__ == "__main__":
    main()
