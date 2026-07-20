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
from torch.utils.data import DataLoader, Dataset, Sampler

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


# ImageNet channel statistics. The 3-channel values are the standard RGB
# constants every ImageNet-pretrained (and MicroNet, which fine-tunes from
# ImageNet) encoder in segmentation_models_pytorch expects. The 1-channel value
# is the standard ImageNet grayscale mean/std, used when a single-channel model
# still wants pretrained-style standardization.
_IMAGENET_MEAN_3 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD_3 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMAGENET_MEAN_1 = np.float32(0.449)
_IMAGENET_STD_1 = np.float32(0.226)


def _apply_channel_norm(
    chw: np.ndarray,
    mode: str,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray:
    """Standardize a CxHxW float32 tile in place-safe fashion, per `mode`.

    "minmax"   -> return unchanged (values stay in [0, 1]).
    "imagenet" -> subtract ImageNet mean and divide by ImageNet std per channel,
                  matching the input distribution ImageNet/MicroNet-pretrained
                  encoders were trained on.
    "dataset"  -> subtract this dataset's own per-channel `mean` and divide by `std`
                  (computed over the training split; see compute_normalization_stats).
                  For from-scratch training where no pretrained distribution applies.

    Must be applied identically at training (dataset.__getitem__) and inference
    (predict_tiles._make_model_input); both call this so they can never drift.
    """
    if mode == "minmax":
        return chw

    channels = chw.shape[0]

    if mode == "imagenet":
        if channels == 3:
            m = _IMAGENET_MEAN_3[:, None, None]
            s = _IMAGENET_STD_3[:, None, None]
        elif channels == 1:
            m = _IMAGENET_MEAN_1
            s = _IMAGENET_STD_1
        else:
            raise ValueError(
                f"imagenet normalization supports 1 or 3 channels, got {channels}."
            )
        return (chw - m) / s

    if mode == "dataset":
        if mean is None or std is None:
            raise ValueError(
                "image_normalization: 'dataset' requires data.norm_mean/norm_std. "
                "Compute them with: python -m fiberseg.tools.compute_dataset_stats "
                "--config <cfg> --write"
            )
        m = np.asarray(mean, dtype=np.float32)
        s = np.asarray(std, dtype=np.float32)
        if m.shape[0] != channels or s.shape[0] != channels:
            raise ValueError(
                f"norm_mean/norm_std length must equal channels={channels}, "
                f"got mean={m.shape[0]}, std={s.shape[0]}."
            )
        return (chw - m[:, None, None]) / s[:, None, None]

    raise ValueError(
        f"Unsupported image_normalization={mode!r}. Use 'minmax', 'imagenet', or 'dataset'."
    )


def compute_normalization_stats(cfg: DataConfig) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over the TRAINING split's percentile-normalized images.

    Uses only the train split (never val/test) so tuning normalization can't leak
    evaluation data. Source images are grayscale, so a single grayscale statistic is
    computed over all training pixels and repeated for `image_channels` (the dataset
    replicates that one grayscale channel, so every channel shares the same stat).
    Computed in the same [0, 1] space `_apply_channel_norm` operates on.
    """
    pairs = [p for p in find_pairs(cfg) if p.split == "train"]
    if not pairs:
        raise ValueError("No training pairs available to compute normalization stats from.")

    total = 0
    pixel_sum = 0.0
    pixel_sqsum = 0.0
    for pair in pairs:
        img = _normalize_image(_read_gray(pair.image_path)).astype(np.float64).ravel()
        total += img.size
        pixel_sum += float(img.sum())
        pixel_sqsum += float(np.square(img).sum())

    mean = pixel_sum / total
    var = max(pixel_sqsum / total - mean * mean, 0.0)
    std = float(np.sqrt(var))
    if std <= 0:
        std = 1.0  # constant image; avoid divide-by-zero at standardization time.

    channels = int(cfg.image_channels)
    return [float(mean)] * channels, [std] * channels


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
        # In "weighted" sampling mode a WeightedTileSampler resamples tiles each
        # epoch, so the dataset must keep ALL tiles (skip the static filter) and
        # return the tile index from __getitem__ so training can attribute per-tile
        # difficulty back for hard-negative mining.
        self.weighted_sampling = split == "train" and cfg.tile_sampling == "weighted"
        self.return_index = self.weighted_sampling
        # Per-tile positive flag (foreground >= min_foreground_fraction), populated
        # by _make_tiles; used by WeightedTileSampler. Empty for non-weighted use.
        self.tile_is_positive: list[bool] = []
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

        # Whether we need each tile's foreground fraction: for static filtering, or
        # to mark positives/negatives for the weighted sampler.
        need_fg = self.split == "train" and (
            self.weighted_sampling
            or self.cfg.min_foreground_fraction > 0
            or self.cfg.keep_empty_probability < 1
        )

        for pair in self.pairs:
            img = _cached_read(str(pair.image_path))
            H, W = img.shape[:2]

            for y in self._positions(H, self.patch_h, self.stride_h):
                for x in self._positions(W, self.patch_w, self.stride_w):
                    fg = None
                    if need_fg:
                        mask = _cached_read(str(pair.mask_path))
                        m = mask[y:y+self.patch_h, x:x+self.patch_w]
                        fg = float((m > 0).mean()) if m.size else 0.0

                    # Static filtering: drop most empty tiles up front. Skipped in
                    # weighted mode, where the sampler handles the balance instead.
                    if not self.weighted_sampling and need_fg and (
                        fg < self.cfg.min_foreground_fraction
                        and rng.random() > self.cfg.keep_empty_probability
                    ):
                        self.filtered_tiles_count += 1
                        continue

                    tiles.append(Tile(pair, y, x, self.patch_h, self.patch_w))
                    if self.weighted_sampling:
                        self.tile_is_positive.append(fg >= self.cfg.min_foreground_fraction)

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

        # Final per-channel standardization (identical to inference). Kept out of
        # the disk cache on purpose: the cache stores percentile-normalized [0, 1]
        # images and stays valid regardless of image_normalization mode.
        img = _apply_channel_norm(
            img.astype(np.float32),
            self.cfg.image_normalization,
            self.cfg.norm_mean,
            self.cfg.norm_std,
        )

        img_t = torch.from_numpy(np.ascontiguousarray(img)).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)

        if self.return_index:
            # 3-tuple only in weighted mode; the training loop uses the index to
            # attribute per-tile difficulty for hard-negative mining.
            return img_t, mask_t, idx

        return img_t, mask_t


class WeightedTileSampler(Sampler):
    """Per-epoch resampling of training tiles for online hard-negative mining.

    Each epoch yields every positive tile plus a fresh draw of
    `round(negative_ratio * n_positives)` negatives. The negative draw is uniform
    during warmup (and whenever hard_negative_fraction == 0); afterward a
    `hard_negative_fraction` slice of it is drawn weighted by each negative tile's
    most recent training loss (hard negatives the model currently gets wrong),
    with the remainder still uniform to preserve coverage.

    `__len__` is constant across epochs (positives + fixed negative count) so
    Lightning's progress bar and step counting stay stable. `update_difficulty`
    is called once per epoch (by HardNegativeMiningCallback) with the losses seen.
    """

    def __init__(
        self,
        is_positive: list[bool],
        negative_ratio: float,
        hard_negative_fraction: float,
        warmup_epochs: int,
        seed: int,
    ):
        flags = np.asarray(is_positive, dtype=bool)
        self.pos = np.flatnonzero(flags)
        self.neg = np.flatnonzero(~flags)
        self.negative_ratio = float(negative_ratio)
        self.hard_fraction = float(hard_negative_fraction)
        self.warmup_epochs = int(warmup_epochs)
        self.seed = int(seed)
        self.epoch = 0
        # Per-tile difficulty (training loss); 1.0 until observed so unseen tiles
        # start with uniform weight.
        self.difficulty = np.ones(flags.shape[0], dtype=np.float64)
        self._n_neg = self._negatives_per_epoch()

    def _negatives_per_epoch(self) -> int:
        if self.pos.size == 0:
            # Degenerate split with no positive tiles: fall back to all negatives.
            return self.neg.size
        want = int(round(self.negative_ratio * self.pos.size))
        return min(self.neg.size, want)

    def update_difficulty(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """Record the latest per-tile training loss (used to weight hard negatives)."""
        self.difficulty[indices] = losses

    def _draw_negatives(self, rng: np.random.Generator) -> np.ndarray:
        n = self._n_neg
        if n == 0 or self.neg.size == 0:
            return np.empty(0, dtype=np.int64)

        uniform = self.epoch < self.warmup_epochs or self.hard_fraction <= 0.0
        if uniform:
            return rng.choice(self.neg, size=n, replace=n > self.neg.size)

        n_hard = int(round(self.hard_fraction * n))
        n_rand = n - n_hard

        weights = np.clip(self.difficulty[self.neg], 1e-6, None)
        weights = weights / weights.sum()
        hard = rng.choice(self.neg, size=n_hard, replace=n_hard > self.neg.size, p=weights)
        rand = rng.choice(self.neg, size=n_rand, replace=n_rand > self.neg.size)
        return np.concatenate([hard, rand])

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.concatenate([self.pos, self._draw_negatives(rng)])
        rng.shuffle(order)
        self.epoch += 1
        return iter(order.tolist())

    def __len__(self) -> int:
        return int(self.pos.size + self._n_neg)


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

            # Build the hard-negative sampler once, if enabled. Held on the
            # datamodule so HardNegativeMiningCallback can push difficulty into it.
            self.train_sampler = None
            if getattr(self.train_ds, "weighted_sampling", False):
                self.train_sampler = WeightedTileSampler(
                    is_positive=self.train_ds.tile_is_positive,
                    negative_ratio=self.cfg.negative_ratio,
                    hard_negative_fraction=self.cfg.hard_negative_fraction,
                    warmup_epochs=self.cfg.hard_negative_warmup_epochs,
                    seed=self.cfg.seed,
                )

        if stage in (None, "test"):
            self.test_ds = TiledSegmentationDataset(self.cfg, "test", self.test_tf)

    def _loader(self, dataset, shuffle: bool, sampler=None):
        cuda_available = torch.cuda.is_available()
        kwargs = {
            "batch_size": self.cfg.batch_size,
            # A custom sampler and shuffle are mutually exclusive in DataLoader.
            "shuffle": shuffle if sampler is None else False,
            "num_workers": self.cfg.num_workers,
            "pin_memory": cuda_available,
            "persistent_workers": self.cfg.num_workers > 0,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler

        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 4

        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        sampler = getattr(self, "train_sampler", None)
        return self._loader(self.train_ds, shuffle=True, sampler=sampler)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)