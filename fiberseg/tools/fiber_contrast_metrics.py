# fiber_contrast_metrics.py
"""Object- or image-level SNR, CNR, and Weber contrast from ground-truth masks.

`compute_fiber_contrast_metrics(..., level="object" | "image")` controls the
granularity: "object" (default) returns one row per fibre; "image" returns
one row per image (aggregated per-object stats plus a separate whole-mask-
pooled computation - see that function's docstring for why "per image" needs
both). For every connected component ("object") in a binary fibre mask, the
object-level computation is:

    SNR   = mu_F / sigma_B
    CNR   = (mu_F - mu_B) / sigma_B
    Weber = (mu_F - mu_B) / mu_B

where `mu_F` is the mean image intensity over the object's fibre pixels, and
`mu_B`/`sigma_B` are the mean/robust-std of a *local* background ring around
that object: the object mask dilated by `dilation_size` px, minus the object
mask dilated by a smaller `gap_size` px (so pixels right at the fibre edge,
which can carry blurred/bled fibre signal, are excluded), minus every fibre
pixel anywhere else in the image (so a neighbouring fibre landing inside the
ring can't contaminate the background estimate). `sigma_B` is the MAD-SD
(1.4826 * median absolute deviation), not the plain standard deviation, so a
stray bright speck or partial neighbour inside the ring doesn't inflate it.

Background is always local (per-object), never a global image statistic,
since SEM images commonly have shading/charging gradients across the field
that would make a single global background mean misleading.

Intensities are read from the *raw* source image (no percentile/contrast
normalization) - CNR is invariant to any affine rescaling of intensity, but
SNR and Weber contrast are not, and the point of these metrics is to
characterize the actual imaging signal, not a training-time normalization
choice.

Run as a script:
    python -m fiberseg.tools.fiber_contrast_metrics \
        --masks-dir data/masks --images-dir data/images --out fiber_contrast.csv

Or import and call directly, e.g. from a notebook:
    from fiberseg.tools.fiber_contrast_metrics import compute_fiber_contrast_metrics
    df = compute_fiber_contrast_metrics("data/masks", "data/images", dilation_size=6)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from ..dataset import IMG_EXTENSIONS, _read_gray

EPS = 1e-8


def mad_sd(values: np.ndarray) -> float:
    """Robust standard deviation via MAD: 1.4826 * median(|x - median(x)|).

    Far less sensitive than plain std() to a handful of outlier pixels (a
    stray bright speck, a sliver of a neighbouring fibre) landing in the
    sample - exactly the failure mode a background-ring estimate is prone to.
    """
    if values.size == 0:
        return 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


def _background_ring(
    obj_mask: np.ndarray,
    full_fibre_mask: np.ndarray,
    dilation_size: int,
    gap_size: int,
) -> np.ndarray:
    """Local background ring for one object: a dilated annulus around it,
    with every fibre pixel in the whole image excluded.

    `gap_size` pixels are skipped immediately outside the object boundary
    (avoids picking up blurred/partial-volume fibre-edge signal as
    "background"), then the ring extends a further `dilation_size` pixels.

    Built from a single Euclidean distance transform rather than two
    disk-footprint dilations: for a *tight* per-object crop the two are
    similar in cost, but `compute_pooled_image_metrics` can end up dilating a
    crop nearly as large as the full image (fibres scattered across most of
    it), where footprint-based dilation with a large disk scales badly
    (~30s on one 5632x8192 image) while the EDT does not (~3s) - and the two
    give pixel-identical rings (disk(r) is exactly the set of pixels within
    Euclidean distance r, which is exactly what the EDT thresholds on).
    """
    dist = ndi.distance_transform_edt(~obj_mask)
    ring = (dist > gap_size) & (dist <= gap_size + dilation_size)
    ring &= ~full_fibre_mask
    return ring


def compute_object_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    dilation_size: int = 5,
    gap_size: int = 2,
    min_object_size: int = 1,
    connectivity: int = 2,
) -> list[dict]:
    """Per-object SNR/CNR/Weber contrast for one image/mask pair.

    `mask > 0` is treated as fibre (this codebase's convention throughout).
    Each connected component (8-connectivity by default, `connectivity=2` -
    appropriate for thin diagonal fibres, which 4-connectivity can fragment)
    is one object. Objects whose background ring ends up empty (e.g. densely
    packed fibres leaving no local background) are skipped, not crashed on.

    Everything below operates on a small crop around each object's bounding
    box (padded by gap_size + dilation_size), not the full image - dilating a
    full-image-sized array per object scales as O(n_objects * H * W), which
    is unusably slow for large SEM images (thousands of px on a side) with
    many separate fibres. `region.image` (regionprops' own tightly-cropped
    local mask) is reused directly instead of a full-image `labeled ==
    region.label` comparison, for the same reason.
    """
    if image.shape != mask.shape:
        raise ValueError(f"image shape {image.shape} does not match mask shape {mask.shape}.")

    binary_mask = mask > 0
    labeled = label(binary_mask, connectivity=connectivity)
    H, W = binary_mask.shape
    pad = gap_size + dilation_size

    results: list[dict] = []
    for region in regionprops(labeled):
        if region.area < min_object_size:
            continue

        min_r, min_c, max_r, max_c = region.bbox
        r0, r1 = max(0, min_r - pad), min(H, max_r + pad)
        c0, c1 = max(0, min_c - pad), min(W, max_c + pad)

        local_obj_mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        local_obj_mask[min_r - r0:max_r - r0, min_c - c0:max_c - c0] = region.image

        local_full_mask = binary_mask[r0:r1, c0:c1]
        local_image = image[r0:r1, c0:c1]

        ring = _background_ring(local_obj_mask, local_full_mask, dilation_size, gap_size)

        fg_values = local_image[local_obj_mask].astype(np.float64)
        bg_values = local_image[ring].astype(np.float64)
        if fg_values.size == 0 or bg_values.size == 0:
            continue

        mu_f = float(fg_values.mean())
        mu_b = float(bg_values.mean())
        sigma_b = mad_sd(bg_values)

        results.append(
            {
                "object_id": int(region.label),
                "area_px": int(region.area),
                "mu_F": mu_f,
                "mu_B": mu_b,
                "sigma_B_mad": sigma_b,
                "n_bg_px": int(bg_values.size),
                "snr": mu_f / (sigma_b + EPS),
                "cnr": (mu_f - mu_b) / (sigma_b + EPS),
                "weber_contrast": (mu_f - mu_b) / (mu_b + EPS),
            }
        )

    return results


def compute_pooled_image_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    dilation_size: int = 5,
    gap_size: int = 2,
) -> dict | None:
    """Whole-image SNR/CNR/Weber contrast: every fibre pixel in the image pooled
    into a single region (no per-fibre connected-component separation), with
    one background ring around that pooled region as a whole.

    This is the more literal "per-image" reading - the image treated as one
    object, exactly parallel to how `compute_object_metrics` treats one fibre
    as one object. It is coarser than aggregating per-object stats
    (`compute_fiber_contrast_metrics(..., level="image")`'s *_mean/*_median
    columns): for images with fibres scattered far apart, the ring can end up
    far from any individual fibre, weakening the "local background" guarantee
    that is the whole point of the per-object design. Returns None if the
    mask has no fibre pixels, or the background ring ends up empty.
    """
    if image.shape != mask.shape:
        raise ValueError(f"image shape {image.shape} does not match mask shape {mask.shape}.")

    binary_mask = mask > 0
    if not binary_mask.any():
        return None

    H, W = binary_mask.shape
    pad = gap_size + dilation_size
    rows_nz, cols_nz = np.nonzero(binary_mask)
    r0, r1 = max(0, int(rows_nz.min()) - pad), min(H, int(rows_nz.max()) + 1 + pad)
    c0, c1 = max(0, int(cols_nz.min()) - pad), min(W, int(cols_nz.max()) + 1 + pad)

    local_obj_mask = binary_mask[r0:r1, c0:c1]
    local_image = image[r0:r1, c0:c1]

    # The pooled region IS the full fibre mask, so it's its own exclusion set -
    # there's no "other" fibre to additionally exclude from the ring.
    ring = _background_ring(local_obj_mask, local_obj_mask, dilation_size, gap_size)

    fg_values = local_image[local_obj_mask].astype(np.float64)
    bg_values = local_image[ring].astype(np.float64)
    if fg_values.size == 0 or bg_values.size == 0:
        return None

    mu_f = float(fg_values.mean())
    mu_b = float(bg_values.mean())
    sigma_b = mad_sd(bg_values)

    return {
        "n_fg_px": int(fg_values.size),
        "mu_F_pooled": mu_f,
        "mu_B_pooled": mu_b,
        "sigma_B_mad_pooled": sigma_b,
        "n_bg_px_pooled": int(bg_values.size),
        "snr_pooled": mu_f / (sigma_b + EPS),
        "cnr_pooled": (mu_f - mu_b) / (sigma_b + EPS),
        "weber_contrast_pooled": (mu_f - mu_b) / (mu_b + EPS),
    }


_OBJECT_COLUMNS = [
    "image", "object_id", "area_px",
    "mu_F", "mu_B", "sigma_B_mad", "n_bg_px",
    "snr", "cnr", "weber_contrast",
]

_IMAGE_LEVEL_COLUMNS = [
    "image", "n_objects", "n_fg_px",
    "snr_mean", "snr_median", "snr_pooled",
    "cnr_mean", "cnr_median", "cnr_pooled",
    "weber_contrast_mean", "weber_contrast_median", "weber_contrast_pooled",
    "mu_F_pooled", "mu_B_pooled", "sigma_B_mad_pooled", "n_bg_px_pooled",
]


def compute_fiber_contrast_metrics(
    masks_dir: str | Path,
    images_dir: str | Path,
    *,
    mask_pattern: str = "{stem}_mask.tif",
    image_glob: str = "*.tif",
    dilation_size: int = 5,
    gap_size: int = 2,
    min_object_size: int = 1,
    level: str = "object",
) -> pd.DataFrame:
    """SNR/CNR/Weber contrast for every mask in `masks_dir`, at object or image level.

    Masks alone don't carry pixel intensities, so `images_dir` (the matching
    source images, paired by `mask_pattern` - same convention as
    `fiberseg.dataset.find_pairs`) is also required.

    `level="object"` (default): one row per fibre object (see
    `compute_object_metrics`) - the finer-grained option, needed for e.g.
    plotting detection recall against per-fibre CNR.

    `level="image"`: one row per image instead. "Per-image SNR" is
    ambiguous, so both real readings are included rather than picking one:
      - `*_mean`/`*_median`: the per-object snr/cnr/weber_contrast values
        (each still computed from that fibre's own local ring) aggregated
        across every fibre in the image.
      - `*_pooled`: every fibre pixel in the image pooled into one region
        with a single ring around the whole mask (see
        `compute_pooled_image_metrics`) - coarser, and less locally-accurate
        for images with fibres scattered far apart, but the more literal
        "treat the image as one object" reading.
    """
    if level not in ("object", "image"):
        raise ValueError(f"level must be 'object' or 'image', got {level!r}.")

    masks_dir = Path(masks_dir)
    images_dir = Path(images_dir)

    image_paths = sorted(
        p for p in images_dir.glob(image_glob)
        if p.suffix.lower() in IMG_EXTENSIONS and not p.stem.endswith("_mask")
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir} using glob {image_glob!r}.")

    rows: list[dict] = []
    pooled_rows: list[dict] = []
    missing: list[str] = []
    skipped_objects = 0

    for img_path in tqdm.tqdm(image_paths, desc="Processing images"):
        mask_name = mask_pattern.format(stem=img_path.stem, suffix=img_path.suffix, name=img_path.name)
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            missing.append(img_path.name)
            continue

        image = _read_gray(img_path)
        mask = _read_gray(mask_path)

        binary_mask = mask > 0
        n_components = label(binary_mask, connectivity=2).max()

        objects = compute_object_metrics(
            image,
            mask,
            dilation_size=dilation_size,
            gap_size=gap_size,
            min_object_size=min_object_size,
        )
        skipped_objects += n_components - len(objects)

        for obj in objects:
            rows.append({"image": img_path.name, **obj})

        if level == "image":
            pooled = compute_pooled_image_metrics(
                image, mask, dilation_size=dilation_size, gap_size=gap_size
            )
            if pooled is not None:
                pooled_rows.append({"image": img_path.name, **pooled})

    if missing:
        print(f"Warning: {len(missing)} image(s) had no matching mask, skipped. First: {missing[0]}")
    if skipped_objects:
        print(
            f"Warning: {skipped_objects} fibre object(s) skipped (below min_object_size or "
            "empty background ring - e.g. densely packed fibres leaving no local background)."
        )

    object_df = pd.DataFrame(rows, columns=_OBJECT_COLUMNS)

    if level == "object":
        return object_df

    if len(object_df):
        agg = object_df.groupby("image", as_index=False).agg(
            n_objects=("object_id", "count"),
            snr_mean=("snr", "mean"), snr_median=("snr", "median"),
            cnr_mean=("cnr", "mean"), cnr_median=("cnr", "median"),
            weber_contrast_mean=("weber_contrast", "mean"),
            weber_contrast_median=("weber_contrast", "median"),
        )
    else:
        agg = pd.DataFrame(columns=["image", "n_objects", "snr_mean", "snr_median",
                                     "cnr_mean", "cnr_median",
                                     "weber_contrast_mean", "weber_contrast_median"])

    pooled_df = pd.DataFrame(
        pooled_rows,
        columns=["image", "n_fg_px", "mu_F_pooled", "mu_B_pooled", "sigma_B_mad_pooled",
                 "n_bg_px_pooled", "snr_pooled", "cnr_pooled", "weber_contrast_pooled"],
    )

    image_df = pd.merge(agg, pooled_df, on="image", how="outer")
    return image_df.reindex(columns=_IMAGE_LEVEL_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Compute object-level (per-fibre) SNR, CNR, and Weber contrast from "
        "ground-truth masks and their source images."
    )
    parser.add_argument("--masks-dir", required=True, help="Folder of ground-truth masks.")
    parser.add_argument("--images-dir", required=True, help="Folder of matching source images.")
    parser.add_argument(
        "--mask-pattern", default="{stem}_mask.tif",
        help="Mask filename pattern, {stem}/{suffix}/{name} placeholders (default: {stem}_mask.tif).",
    )
    parser.add_argument("--image-glob", default="*.tif", help="Glob for source images (default: *.tif).")
    parser.add_argument(
        "--dilation-size", type=int, default=5,
        help="Background ring outer thickness in pixels, beyond --gap-size (default: 5).",
    )
    parser.add_argument(
        "--gap-size", type=int, default=2,
        help="Pixels skipped immediately outside each fibre before the background ring "
        "starts, to avoid bleeding fibre-edge signal into the background estimate (default: 2).",
    )
    parser.add_argument(
        "--min-object-size", type=int, default=1,
        help="Discard connected components smaller than this many pixels (default: 1, i.e. no filtering).",
    )
    parser.add_argument(
        "--level", choices=["object", "image"], default="object",
        help="'object' (default): one row per fibre. 'image': one row per image, with "
        "*_mean/*_median (aggregated per-object stats) and *_pooled (whole mask treated "
        "as one region) columns.",
    )
    parser.add_argument("--out", default="fiber_contrast_metrics.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = compute_fiber_contrast_metrics(
        args.masks_dir,
        args.images_dir,
        mask_pattern=args.mask_pattern,
        image_glob=args.image_glob,
        dilation_size=args.dilation_size,
        gap_size=args.gap_size,
        min_object_size=args.min_object_size,
        level=args.level,
    )

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    if args.level == "object":
        print(f"Wrote {len(df)} fibre object(s) from {df['image'].nunique()} image(s) to {out_path}")
        if len(df):
            print(
                "Median snr={:.2f}, cnr={:.2f}, weber_contrast={:.2f}".format(
                    df["snr"].median(), df["cnr"].median(), df["weber_contrast"].median()
                )
            )
    else:
        print(f"Wrote {len(df)} image(s) to {out_path}")
        if len(df):
            print(
                "Median (per-object mean) snr={:.2f}, cnr={:.2f}, weber_contrast={:.2f}".format(
                    df["snr_mean"].median(), df["cnr_mean"].median(), df["weber_contrast_mean"].median()
                )
            )


if __name__ == "__main__":
    main()
