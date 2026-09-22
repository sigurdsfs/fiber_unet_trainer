# standardize_pixel_size.py
"""Resample an image/mask dataset to one common physical pixel size (nm/px).

The training set mixes acquisitions from ~14 to ~90 nm/px, so the same fiber is up to
~6x wider in pixels in one image than another - far more than the Affine scale
augmentation covers - and the model has to learn every scale at once. Resampling all
images (and their masks) to one nm/px removes that nuisance variable.

Non-destructive: the source folders are only read. Output goes to a NEW folder:

    <out-dir>/Images/<same filename>        resampled image (same dtype, zlib TIFF)
    <out-dir>/Masks/<same filename>         resampled mask  (same dtype / fg value)
    <out-dir>/pixel_size_manifest.csv       one row per image: source nm/px, where it came
                                            from, scale factor, sizes, fg fraction before/after
    <out-dir>/unresolved_pixel_sizes.xlsx   review sheet for the images with no known pixel
                                            size - fill in `pixel_size_nm` and re-run with
                                            --overrides <that file>

Filenames are kept identical and (by default) EVERY image/mask pair is written, so
`dataset.find_pairs` - which shuffles the sorted pair list with `data.seed` - produces
exactly the same train/val/test split on the new folder as on the original. Pointing a
config's data.images_dir/masks_dir at <out-dir>/Images and <out-dir>/Masks is therefore a
one-variable change. Images whose pixel size can't be resolved are copied UNSCALED by
default for that reason (flagged `unresolved_copied` in the manifest); `--unresolved skip`
drops them instead, which changes the split.

Pixel size comes from `tools.pixel_size.pixel_size_nm` (vendor metadata, then filename),
overridden per image by `--overrides` (CSV/XLSX with `image` + `pixel_size_nm` columns,
e.g. the review sheet above or the one `tools.pixel_size --out x.xlsx` writes). Each
written TIFF carries its new pixel size as `<PixelWidth_um>` in ImageDescription, which
`pixel_size_nm` reads back.

Masks: resampled as fiber *coverage* (area-average when shrinking, bilinear when
enlarging) and re-binarized. When shrinking by factor f the coverage cut is 0.5/f
instead of 0.5, so a 1-px-wide fiber in the source (coverage ~1/f after shrinking)
survives instead of vanishing - at the cost of up to ~half an output pixel of widening
on fiber edges. `--mask-min-coverage` overrides it.

Inference must see the same scale: images fed to a model trained on this output have to
be resampled to the same --target-nm first (already true for acquisitions at that nm/px).

Run:
    python -m fiberseg.tools.standardize_pixel_size \
        --images-dir "../Current Training Material - fibers only - Preproc/Images" \
        --masks-dir  "../Current Training Material - fibers only - Preproc/Masks" \
        --out-dir    "../Current Training Material - fibers only - Preproc/PixelSize_50nm" \
        --target-nm 50
Add --dry-run to only resolve pixel sizes and write the manifest.
"""
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import tqdm

from ..dataset import IMG_EXTENSIONS, _read_gray
from .pixel_size import pixel_size_nm, write_review_xlsx

# Below this relative scale change the file is copied byte-for-byte instead of resampled:
# a <0.5% resize only adds interpolation blur for no real change in fiber width.
_COPY_TOLERANCE = 0.005


@dataclass
class ResolvedSize:
    nm: float | None
    source: str
    error: str = ""


def load_overrides(path: str | Path | None) -> dict[str, float]:
    """`image` -> `pixel_size_nm` from a CSV/XLSX; blank rows are ignored."""
    if not path:
        return {}
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    missing = {"image", "pixel_size_nm"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    df = df.dropna(subset=["image", "pixel_size_nm"])
    return {str(r.image).strip(): float(r.pixel_size_nm) for r in df.itertuples()}


def resolve_size(image_path: Path, overrides: dict[str, float]) -> ResolvedSize:
    if image_path.name in overrides:
        return ResolvedSize(overrides[image_path.name], "override")
    try:
        ps = pixel_size_nm(str(image_path))
        return ResolvedSize(ps.nm, ps.source)
    except Exception as exc:  # any failure just means "unresolved"
        return ResolvedSize(None, "", str(exc))


def _out_shape(h: int, w: int, scale: float) -> tuple[int, int]:
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def resample_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize by `scale` (output/input), preserving dtype. Area-average when shrinking
    (anti-aliased), bicubic when enlarging."""
    h, w = img.shape[:2]
    oh, ow = _out_shape(h, w, scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    out = cv2.resize(img.astype(np.float32), (ow, oh), interpolation=interp)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(img.dtype)


def mask_coverage_threshold(scale: float) -> float:
    """0.5 when enlarging; 0.5/f when shrinking by f = 1/scale (see module docstring)."""
    return 0.5 if scale >= 1.0 else 0.5 * scale


def resample_mask(
    mask: np.ndarray, scale: float, min_coverage: float | None = None
) -> np.ndarray:
    """Resize a binary mask (fg = `mask > 0`) by `scale`, returning the same dtype with
    foreground set to the input's max value."""
    fg = (mask > 0).astype(np.float32)
    h, w = mask.shape[:2]
    oh, ow = _out_shape(h, w, scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    coverage = cv2.resize(fg, (ow, oh), interpolation=interp)
    cut = mask_coverage_threshold(scale) if min_coverage is None else min_coverage
    fg_value = mask.max() if mask.max() > 0 else (
        np.iinfo(mask.dtype).max if np.issubdtype(mask.dtype, np.integer) else 1
    )
    out = np.zeros((oh, ow), dtype=mask.dtype)
    out[coverage >= cut - 1e-6] = fg_value
    return out


def _read_raw(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    return _read_gray(path)


def _write(path: Path, arr: np.ndarray, pixel_nm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".tif", ".tiff"):
        # Same tag convention pixel_size._from_xml_tags reads, so the new calibration is
        # recoverable from the file itself.
        desc = f"<PixelWidth_um>{pixel_nm / 1000.0:.9g}</PixelWidth_um>"
        tifffile.imwrite(path, arr, compression="zlib", description=desc, metadata=None)
    else:
        cv2.imwrite(str(path), arr)


def find_image_mask_pairs(
    images_dir: Path, masks_dir: Path, image_glob: str, mask_pattern: str
) -> list[tuple[Path, Path | None]]:
    """Same discovery rule as `dataset.find_pairs` (glob, drop `*_mask` stems, format
    `mask_pattern`), but keeps images with no mask (mask=None) instead of dropping them."""
    images = sorted(
        p for p in images_dir.glob(image_glob)
        if p.suffix.lower() in IMG_EXTENSIONS and not p.stem.endswith("_mask")
    )
    out = []
    for img in images:
        mask = masks_dir / mask_pattern.format(stem=img.stem, suffix=img.suffix, name=img.name)
        out.append((img, mask if mask.exists() else None))
    return out


def standardize(
    images_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    target_nm: float,
    *,
    image_glob: str = "*.tif",
    mask_pattern: str = "{stem}_mask.tif",
    overrides: dict[str, float] | None = None,
    unresolved: str = "copy",
    mask_min_coverage: float | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    if unresolved not in ("copy", "skip"):
        raise ValueError("unresolved must be 'copy' or 'skip'")
    images_dir, masks_dir, out_dir = Path(images_dir), Path(masks_dir), Path(out_dir)
    if out_dir.resolve() in (images_dir.resolve(), masks_dir.resolve()):
        raise ValueError("--out-dir must differ from the source folders (originals are kept).")
    out_images, out_masks = out_dir / "Images", out_dir / "Masks"
    overrides = overrides or {}

    pairs = find_image_mask_pairs(images_dir, masks_dir, image_glob, mask_pattern)
    if not pairs:
        raise FileNotFoundError(f"No images matched {image_glob!r} in {images_dir}")

    rows = []
    for img_path, mask_path in tqdm.tqdm(pairs, desc="Standardizing pixel size"):
        size = resolve_size(img_path, overrides)
        row = {
            "image": img_path.name,
            "mask": mask_path.name if mask_path else "",
            "pixel_size_nm": size.nm if size.nm is not None else math.nan,
            "source": size.source,
            "error": size.error,
            "target_nm": target_nm,
        }
        if size.nm is None:
            status, scale = ("unresolved_copied" if unresolved == "copy" else "skipped"), 1.0
        elif abs(size.nm / target_nm - 1.0) <= _COPY_TOLERANCE:
            status, scale = "copied_at_target", 1.0
        else:
            status, scale = "resampled", size.nm / target_nm
        row.update(status=status, scale=scale)

        dst_img = out_images / img_path.name
        dst_mask = out_masks / mask_path.name if mask_path else None
        done = dst_img.exists() and (dst_mask is None or dst_mask.exists())
        if dry_run or status == "skipped" or (done and not overwrite):
            if done and not overwrite and not dry_run and status != "skipped":
                row["status"] = status + " (existing, not rewritten)"
            rows.append(row)
            continue

        img = _read_raw(img_path)
        row["width_px"], row["height_px"] = img.shape[1], img.shape[0]
        mask = _read_raw(mask_path) if mask_path else None
        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                raise ValueError(
                    f"{mask_path.name} shape {mask.shape[:2]} != image {img.shape[:2]}"
                )
            row["gt_fg_fraction_before"] = float((mask > 0).mean())

        if scale == 1.0:
            # Byte-exact copy; unresolved/at-target files are not re-encoded.
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)
            if mask_path:
                dst_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, dst_mask)
            new_img, new_mask = img, mask
        else:
            new_img = resample_image(img, scale)
            _write(dst_img, new_img, target_nm)
            new_mask = None
            if mask is not None:
                new_mask = resample_mask(mask, scale, mask_min_coverage)
                _write(dst_mask, new_mask, target_nm)

        row["out_width_px"], row["out_height_px"] = new_img.shape[1], new_img.shape[0]
        if new_mask is not None:
            row["gt_fg_fraction_after"] = float((new_mask > 0).mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pixel_size_manifest.csv", index=False)

    unresolved_df = df[df["pixel_size_nm"].isna()]
    review_path = out_dir / "unresolved_pixel_sizes.xlsx"
    if not len(unresolved_df) and review_path.exists():
        review_path.unlink()  # stale sheet from an earlier run that had unresolved images
    if len(unresolved_df):
        review = pd.DataFrame({
            "image": unresolved_df["image"],
            "pixel_size_nm": math.nan,
            "source": "",
            "width_px": unresolved_df.get("width_px"),
            "height_px": unresolved_df.get("height_px"),
            "error": unresolved_df["error"],
            "path": [str(images_dir / n) for n in unresolved_df["image"]],
        })
        write_review_xlsx(review, review_path)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Resample images and masks to one common nm/px into a new folder "
        "(originals untouched; filenames and split preserved)."
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="New folder; gets Images/ and Masks/.")
    parser.add_argument(
        "--target-nm", type=float, required=True,
        help="Common pixel size in nm/px. Pick the acquisition setting the model will be "
        "deployed on, so production images need no resampling.",
    )
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--mask-pattern", default="{stem}_mask.tif")
    parser.add_argument(
        "--overrides", default=None,
        help="CSV/XLSX with `image` and `pixel_size_nm` columns; wins over auto-detection.",
    )
    parser.add_argument(
        "--unresolved", choices=["copy", "skip"], default="copy",
        help="Images with no known pixel size: copy unscaled (default; keeps the split "
        "identical) or skip (changes the split).",
    )
    parser.add_argument(
        "--mask-min-coverage", type=float, default=None,
        help="Override the fiber-coverage cut used to re-binarize masks "
        "(default 0.5 enlarging, 0.5/f shrinking by f).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing outputs.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only resolve pixel sizes and write the manifest; no images written.",
    )
    args = parser.parse_args()

    df = standardize(
        Path(args.images_dir), Path(args.masks_dir), Path(args.out_dir), args.target_nm,
        image_glob=args.image_glob, mask_pattern=args.mask_pattern,
        overrides=load_overrides(args.overrides), unresolved=args.unresolved,
        mask_min_coverage=args.mask_min_coverage, overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print(f"\n{len(df)} images -> {args.out_dir}")
    print(df["status"].value_counts().to_string())
    known = df["pixel_size_nm"].dropna()
    if len(known):
        print(
            f"Source pixel size (nm/px): min {known.min():.3g}, median {known.median():.3g}, "
            f"max {known.max():.3g}; scale factors {df.loc[known.index, 'scale'].min():.3g}"
            f"-{df.loc[known.index, 'scale'].max():.3g}"
        )
    n_unres = int(df["pixel_size_nm"].isna().sum())
    if n_unres:
        print(
            f"WARNING: {n_unres} images have no known pixel size and were "
            f"{'copied UNSCALED' if args.unresolved == 'copy' else 'skipped'}. Fill in "
            f"{Path(args.out_dir) / 'unresolved_pixel_sizes.xlsx'} and re-run with "
            "--overrides <that file> --overwrite."
        )


if __name__ == "__main__":
    main()
