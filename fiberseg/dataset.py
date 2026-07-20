# dataset.py
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augmentations import build_transform
from .config import DataConfig

IMG_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    mask_path: Path
    split: str


@dataclass(frozen=True)
class Tile:
    pair: Pair
    y: int
    x: int
    h: int
    w: int


def _hw(value: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("patch_size/stride must be an int or [height, width].")
    return int(value[0]), int(value[1])


def _read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path).convert("L"))
    if arr.ndim == 3:
        # If RGB-like, convert to grayscale by averaging channels.
        # SEM data is normally already grayscale.
        arr = arr[..., :3].mean(axis=-1)
    return arr


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert SEM image to float32 and robustly normalize to [0, 1].

    Works for uint8, uint16, and floating-point images.
    Uses percentile normalization so a few very bright pixels
    do not dominate the scaling.
    """
    img = img.astype(np.float32)

    if img.size == 0:
        return img

    # Protect against NaN or infinite values, just in case.
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Robust contrast normalization.
    lo, hi = np.percentile(img, [1.0, 99.5])

    # Fallback for nearly constant images.
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())

    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _pad_to_shape(arr: np.ndarray, h: int, w: int, constant: float = 0) -> np.ndarray:
    pad_h = max(0, h - arr.shape[0])
    pad_w = max(0, w - arr.shape[1])
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=constant)


def find_pairs(cfg: DataConfig) -> list[Pair]:
    images_dir = Path(cfg.images_dir)
    masks_dir = Path(cfg.masks_dir)

    # Find candidate image files.
    # Important: exclude ground-truth masks ending in "_binary".
    image_paths = sorted(
        p for p in images_dir.glob(cfg.image_glob)
        if p.suffix.lower() in IMG_EXTENSIONS
        and not p.stem.endswith("_mask")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No non-binary input images found in {images_dir} using glob {cfg.image_glob!r}. "
            "Files ending in '_binary' are treated as ground-truth masks and excluded."
        )

    pairs_raw: list[tuple[Path, Path]] = []
    missing: list[Path] = []

    for img in image_paths:
        # Example:
        # image: sample_001.tif
        # mask:  sample_001_binary.tif
        mask_name = cfg.mask_pattern.format(
            stem=img.stem,
            suffix=img.suffix,
            name=img.name,
        )
        mask = masks_dir / mask_name

        if mask.exists():
            pairs_raw.append((img, mask))
        else:
            missing.append(mask)

    if not pairs_raw:
        msg = "No image/mask pairs found. Check masks_dir and mask_pattern."
        if missing:
            msg += f" First expected mask was: {missing[0]}"
        raise FileNotFoundError(msg)

    if missing:
        print(
            f"Warning: found {len(pairs_raw)} image/mask pairs, "
            f"but {len(missing)} images had no matching mask. "
            f"First missing mask: {missing[0]}"
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs_raw)

    n = len(pairs_raw)
    n_train = int(round(n * cfg.split.get("train", 0.7)))
    n_val = int(round(n * cfg.split.get("val", 0.15)))

    pairs: list[Pair] = []

    for i, (img, mask) in enumerate(pairs_raw):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        pairs.append(Pair(img, mask, split))

    return pairs


def split_filenames(cfg: DataConfig) -> dict[str, list[str]]:
    """Group input image filenames by split ("Train"/"validation"/"Test").

    Reproduces `find_pairs`'s deterministic seed+fraction split, so results only match a
    given training run if `images_dir` still contains the same files it did at training time.
    """
    key_by_split = {"train": "Train", "val": "validation", "test": "Test"}
    result: dict[str, list[str]] = {"Train": [], "validation": [], "Test": []}

    for pair in find_pairs(cfg):
        result[key_by_split[pair.split]].append(pair.image_path.name)

    return result


@lru_cache(maxsize=16)
def _cached_read(path_str: str) -> np.ndarray:
    return _read_gray(Path(path_str))


@lru_cache(maxsize=16)
def _cached_normalized_image(path_str: str) -> np.ndarray:
    """Percentile-normalize a source image once, at full-image resolution.

    Tiles must be cropped from this pre-normalized array rather than each
    calling `_normalize_image` on its own crop: per-tile percentiles would
    give every tile its own contrast stretch and drift from the whole-image
    normalization `predict_tiles.py` uses at inference time.
    """
    return _normalize_image(_cached_read(path_str))


def _resolve_cache_dir(cfg: DataConfig) -> Path:
    return Path(cfg.cache_dir) if cfg.cache_dir else Path("cache") / "normalized_images"


def _cache_file_path(path: Path, cache_dir: Path, tag: str) -> Path:
    stat = path.stat()
    raw_key = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{tag}"
    digest = hashlib.sha1(raw_key.encode()).hexdigest()[:16]
    return cache_dir / f"{path.stem}_{tag}_{digest}.npy"


def _load_or_build_disk_cache(path: Path, cache_dir: Path, tag: str, build_fn) -> np.ndarray:
    """Compute `build_fn(path)` once and persist it to disk as .npy, then return a
    memory-mapped read-only view.

    Unlike an in-process `lru_cache`, this cache survives across epochs, training
    runs, and sweep combinations (as long as the source file doesn't change), and
    the underlying pixel data is shared across DataLoader worker processes via the
    OS page cache instead of being duplicated in each worker's own memory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_path(path, cache_dir, tag)
    if not cache_path.exists():
        arr = build_fn(path)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp{os.getpid()}")
        # Pass an open file handle, not a path string: np.save silently appends
        # ".npy" to string paths that don't already end in it, which would break
        # the rename below.
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        try:
            os.replace(tmp_path, cache_path)
        except OSError:
            # Another worker process built and opened this same cache entry first
            # (Windows can refuse to replace a file another process has
            # memory-mapped). Discard our redundant copy and load theirs instead.
            tmp_path.unlink(missing_ok=True)

    # Opening the file right after another process just created/renamed it can
    # transiently fail on Windows (e.g. antivirus briefly holding an exclusive
    # handle on the new file). Retry with backoff instead of treating it as fatal.
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            return np.load(cache_path, mmap_mode="r")
        except (PermissionError, OSError):
            if attempt == max_attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise AssertionError("unreachable")


@lru_cache(maxsize=512)
def _mmap_normalized_image(path_str: str, cache_dir_str: str) -> np.ndarray:
    return _load_or_build_disk_cache(
        Path(path_str), Path(cache_dir_str), "norm", lambda p: _normalize_image(_read_gray(p))
    )


@lru_cache(maxsize=512)
def _mmap_mask(path_str: str, cache_dir_str: str) -> np.ndarray:
    return _load_or_build_disk_cache(Path(path_str), Path(cache_dir_str), "mask", _read_gray)


class TiledSegmentationDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str, augmentations=None):
        self.cfg = cfg
        self.split = split
        self.patch_h, self.patch_w = _hw(cfg.patch_size)
        self.stride_h, self.stride_w = _hw(cfg.stride if cfg.stride is not None else cfg.patch_size)
        self.transform = augmentations
        self.pairs = [p for p in find_pairs(cfg) if p.split == split]
        if not self.pairs:
            raise ValueError(
                f"No pairs for split={split!r}. Add more images or adjust split fractions."
            )
        self.filtered_tiles_count = 0
        self.tiles = self._make_tiles()
        if not self.tiles:
            raise ValueError(f"No tiles found for split={split!r}.")

    def _positions(self, size: int, patch: int, stride: int) -> list[int]:
        if size <= patch:
            return [0]
        pos = list(range(0, size - patch + 1, stride))
        if pos[-1] != size - patch:
            pos.append(size - patch)
        return pos

    def _make_tiles(self) -> list[Tile]:
        split_offset = {
            "train": 0,
            "val": 1000,
            "test": 2000,
        }.get(self.split, 3000)

        rng = random.Random(self.cfg.seed + split_offset)

        tiles: list[Tile] = []

        for pair in self.pairs:
            img = _cached_read(str(pair.image_path))
            H, W = img.shape[:2]

            for y in self._positions(H, self.patch_h, self.stride_h):
                for x in self._positions(W, self.patch_w, self.stride_w):
                    if self.split == "train" and (
                        self.cfg.min_foreground_fraction > 0
                        or self.cfg.keep_empty_probability < 1
                    ):
                        mask = _cached_read(str(pair.mask_path))
                        m = mask[y:y+self.patch_h, x:x+self.patch_w]
                        fg = float((m > 0).mean()) if m.size else 0.0 

                        if (
                            fg < self.cfg.min_foreground_fraction
                            and rng.random() > self.cfg.keep_empty_probability
                        ):
                            self.filtered_tiles_count += 1
                            continue

                    tiles.append(Tile(pair, y, x, self.patch_h, self.patch_w))

        return tiles

    @property
    def filtered_tiles_fraction(self) -> float:
        total_tiles = len(self.tiles) + self.filtered_tiles_count
        if total_tiles == 0:
            return 0.0
        return self.filtered_tiles_count / total_tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int):
        tile = self.tiles[idx]

        cache_dir_str = str(_resolve_cache_dir(self.cfg))
        norm_img = _mmap_normalized_image(str(tile.pair.image_path), cache_dir_str)
        mask_full = _mmap_mask(str(tile.pair.mask_path), cache_dir_str)

        # Materialize out of the memmap immediately: augmentations need a
        # writable, contiguous array, not a read-only strided view.
        img = np.array(norm_img[tile.y:tile.y+tile.h, tile.x:tile.x+tile.w])
        mask = np.array(mask_full[tile.y:tile.y+tile.h, tile.x:tile.x+tile.w])

        img = _pad_to_shape(img, tile.h, tile.w, constant=0)
        mask = _pad_to_shape(mask, tile.h, tile.w, constant=0)

        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        # Augmentations (brightness/contrast, noise, blur, interpolation) can push
        # values outside the already-normalized [0, 1] range. Clip as the final
        # step so every tensor fed to the model is guaranteed to be in [0, 1].
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        if self.cfg.image_channels == 1:
            img = img[None, :, :]
        elif self.cfg.image_channels == 3:
            img = np.stack([img, img, img], axis=0)
        else:
            raise ValueError(
                f"Unsupported image_channels={self.cfg.image_channels}. "
                "Use image_channels: 1 or image_channels: 3."
            )

        img_t = torch.from_numpy(np.ascontiguousarray(img)).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)

        return img_t, mask_t


class FiberDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig, augmentations: dict | None = None):
        super().__init__()
        self.cfg = cfg
        augmentations = augmentations or {}
        self.train_tf = build_transform(augmentations.get("train"))
        self.val_tf = build_transform(augmentations.get("val"))
        self.test_tf = build_transform(augmentations.get("test"))

    @property
    def filtered_tiles_count(self) -> int:
        return getattr(self.train_ds, "filtered_tiles_count", 0)

    @property
    def filtered_tiles_fraction(self) -> float:
        return getattr(self.train_ds, "filtered_tiles_fraction", 0.0)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = TiledSegmentationDataset(self.cfg, "train", self.train_tf)
            self.val_ds = TiledSegmentationDataset(self.cfg, "val", self.val_tf)

        if stage in (None, "test"):
            self.test_ds = TiledSegmentationDataset(self.cfg, "test", self.test_tf)

    def _loader(self, dataset, shuffle: bool):
        cuda_available = torch.cuda.is_available()
        kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": self.cfg.num_workers,
            "pin_memory": cuda_available,
            "persistent_workers": self.cfg.num_workers > 0,
        }

        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 4

        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)