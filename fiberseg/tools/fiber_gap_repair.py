# fiber_gap_repair.py
"""Repair fragmentation ("holes") in thin elongated fibre masks/predictions.

Length and aspect-ratio-based WHO/ISO fibre counting is only valid on
reconstructed *whole* fibres - a contrast dip, occlusion, or thin
low-confidence region along a fibre commonly drops below threshold and
splits one physical fibre into several mask fragments. This module
implements two families of technique for that failure mode, matching the
two points in the pipeline where it can be addressed:

**Thresholding-time (need the continuous probability map, not just the
final binary mask)** - use these to build a *better starting mask* before
it ever gets fragmented:

- `hysteresis_threshold_mask()` - Canny-style two-threshold hysteresis:
  a high threshold seeds confident fibre cores, a low threshold grows
  outward only where connected to a seed, keeping thin low-confidence
  continuations a single hard threshold would sever.
- `enhance_ridges()` - Frangi/Sato (Hessian-based) ridge/vesselness
  enhancement, the same filter family used for vessel/neurite enhancement,
  which amplifies elongated structures and tends to close small gaps along
  the fibre axis before thresholding.

**Post-hoc (work on an already-binarized mask - ground truth or a saved
prediction)** - use `repair_fiber_gaps()`, which combines:

- Oriented morphological closing (`oriented_closing()`): closes along each
  fibre's own local orientation (from the structure tensor) instead of
  isotropically, so gaps bridge along the fibre without fattening it or
  fusing distinct neighbours.
- Skeleton endpoint detection + a collinearity gate (`skeleton_endpoints`):
  candidate fragment-to-fragment links are only accepted when both
  fragments' local tangents point toward each other within `max_angle_deg`
  - this is what prevents two distinct fibres that happen to pass close to
  each other from being wrongly merged into one (a counting error).
- Geodesic (minimum-cost-path) linking (`route_through_array`-based):
  gated candidate pairs are connected via the minimum-cost path through a
  `1 - probability` cost field when a probability map is available (falls
  back to a straight line otherwise), so the connector prefers whatever
  supporting evidence exists in the gap rather than cutting through empty
  background.
- Global, cost-ranked fragment merging: every gated candidate link across
  the whole image is collected, sorted by cost (gap distance + angular
  deviation, optionally + path cost), and accepted greedily with each
  fragment endpoint usable at most once - giving global control over the
  precision/recall trade-off rather than greedy local-only decisions.

Run as a script:
    python -m fiberseg.tools.fiber_gap_repair \
        --config <cfg> --pred-dir <predict_all.py out-dir> --out-dir repaired/

Or import and call directly, e.g. from a notebook:
    from fiberseg.tools.fiber_gap_repair import repair_fiber_gaps
    repaired = repair_fiber_gaps(mask, probability=prob_map, max_gap=15, max_angle_deg=30)
"""
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.draw import line as draw_line
from skimage.filters import apply_hysteresis_threshold, frangi, sato
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from skimage.morphology import (
    dilation,
    disk,
    remove_small_objects,
    skeletonize,
)

from ..config import load_config
from ..dataset import SPLIT_DIRS, _read_gray, find_pairs
from ..predict_tiles import save_mask
from .evaluate_predictions import FIELDNAMES, compute_metrics

EPS = 1e-8
METRIC_NAMES = FIELDNAMES[2:]
COMBINED_FIELDNAMES = (
    ["image", "split"]
    + [f"raw_{m}" for m in METRIC_NAMES]
    + [f"repaired_{m}" for m in METRIC_NAMES]
    + [f"delta_{m}" for m in METRIC_NAMES]
)

# ---------------------------------------------------------------------------
# 1. Hysteresis thresholding (probability map -> binary mask)
# ---------------------------------------------------------------------------


def hysteresis_threshold_mask(probability: np.ndarray, low: float, high: float) -> np.ndarray:
    """Canny-style two-threshold hysteresis on a [0,1] probability map.

    Pixels >= `high` seed confident fibre cores; pixels >= `low` are kept
    only where connected (8-connectivity) to a seed, so thin low-confidence
    continuations of a real fibre survive while isolated low-confidence
    noise does not. `low` must be < `high`.
    """
    if not low < high:
        raise ValueError(f"low ({low}) must be < high ({high}).")
    return apply_hysteresis_threshold(probability, low, high)


# ---------------------------------------------------------------------------
# 2. Ridge / vesselness enhancement (probability map or raw image -> float map)
# ---------------------------------------------------------------------------


def enhance_ridges(
    image: np.ndarray,
    *,
    method: str = "frangi",
    sigmas: tuple[float, ...] = (1, 2, 3),
) -> np.ndarray:
    """Frangi or Sato (Hessian-based) ridge enhancement.

    Amplifies elongated ridge-like structures (fibres are bright ridges on
    the probability map, hence `black_ridges=False`) and tends to bridge
    small gaps along the ridge axis. Intended as a pre-thresholding step:
    run this on the raw probability map, then threshold the result (plain
    or via `hysteresis_threshold_mask`) instead of thresholding the raw
    probability map directly. Output is not normalized to [0,1].
    """
    if method not in ("frangi", "sato"):
        raise ValueError(f"method must be 'frangi' or 'sato', got {method!r}.")
    image = image.astype(np.float64)
    if method == "frangi":
        return frangi(image, sigmas=sigmas, black_ridges=False)
    return sato(image, sigmas=sigmas, black_ridges=False)


# ---------------------------------------------------------------------------
# 3. Oriented morphological closing
# ---------------------------------------------------------------------------


def estimate_orientation_field(field: np.ndarray, *, sigma: float = 2.0) -> np.ndarray:
    """Local fibre-axis orientation (radians, in (-pi/2, pi/2]) from the structure tensor.

    `field` is typically the mask (as float) or a probability map. The
    structure tensor's dominant eigenvector points along the direction of
    maximum local intensity change (the fibre's *width* direction, i.e. the
    edge normal); the fibre axis is perpendicular to that.
    """
    from skimage.feature import structure_tensor

    field = field.astype(np.float64)
    # structure_tensor(..., order="rc") returns (Arr, Arc, Acc) - row-row,
    # row-col, col-col. The standard eigenvector-angle formula is defined in
    # (x, y) = (col, row) terms: Ixx = Acc, Iyy = Arr, Ixy = Arc.
    a_rr, a_rc, a_cc = structure_tensor(field, sigma=sigma, order="rc")
    # Edge-normal orientation from the structure tensor; +pi/2 rotates to
    # the ridge/fibre-axis orientation.
    normal_angle = 0.5 * np.arctan2(2.0 * a_rc, a_cc - a_rr)
    axis_angle = normal_angle + np.pi / 2
    # Wrap to (-pi/2, pi/2]; orientation is undirected (mod pi).
    axis_angle = np.mod(axis_angle + np.pi / 2, np.pi) - np.pi / 2
    return axis_angle


def _line_footprint(radius: int, angle: float, width: float = 0.75) -> np.ndarray:
    """A filled, elongated rectangular footprint of half-length `radius` and
    half-width `width`, at `angle` radians (0 = horizontal/along columns,
    matching (row, col) image convention).

    Built from a direct inequality test on rotated coordinates rather than
    rasterizing a thin (1px) rotated line: a 1px rotated line becomes a
    staircase at non-axis-aligned angles, which is not reliably 8-connected.
    `width` is kept close to its minimum safe value (sqrt(2)/2 =~ 0.707,
    guaranteeing 8-connectivity at any angle) rather than made generously
    thick, because - since `oriented_closing` uses this for plain dilation,
    not full closing (see there for why) - any extra width also thickens the
    fibre along its *entire* length wherever it's used, not just at gaps.
    """
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    along = yy * sin_a + xx * cos_a
    cross = -yy * cos_a + xx * sin_a
    return (np.abs(along) <= radius) & (np.abs(cross) <= width)


def oriented_closing(
    mask: np.ndarray,
    *,
    radius: int = 4,
    n_bins: int = 8,
    orientation_sigma: float = 2.0,
) -> np.ndarray:
    """Directional gap-bridging: extends each fibre along its own local
    orientation instead of isotropically, so short gaps bridge along the
    fibre without fattening it or fusing nearby fibres running in a
    different direction the way a plain disk-based closing would.

    Discretizes the structure-tensor orientation field into `n_bins` angle
    bins; for each bin, dilates the whole mask with a filled, elongated
    footprint at that angle, but only keeps newly-added pixels that fall
    near existing mask pixels already assigned to that same orientation bin.
    Dilation (not full closing - see `_line_footprint`) is used deliberately;
    since every newly-added pixel is already gated to be near a same-oriented
    fibre, the one thing this does *not* guard against is a fibre endpoint
    with no real partner nearby still extending outward by up to `radius`
    pixels along its own direction - a small, bounded, direction-consistent
    edge effect, not a fabricated cross-fibre connection. Keep `radius`
    modest (the default) to bound it; the collinearity-gated linking in
    `repair_fiber_gaps` is what actually decides whether two fragments
    belong together.
    """
    if not mask.any():
        return mask.copy()

    orientation = estimate_orientation_field(mask.astype(np.float64), sigma=orientation_sigma)
    # Bin centers land on exact multiples of the bin width, starting at 0 -
    # e.g. for n_bins=8: 0, +-22.5, +-45, +-67.5, 90 degrees - so an exactly
    # axis-aligned fibre (very common in synthetic/regular cases, and the
    # common case generally) is assigned a footprint at its *exact* angle
    # instead of being split across two neighbouring bins by a boundary that
    # happens to fall right on 0.
    bin_width = np.pi / n_bins
    bin_centers = np.arange(n_bins) * bin_width - np.pi / 2
    offset = np.round((orientation - (-np.pi / 2)) / bin_width).astype(int)
    bin_idx = np.mod(offset, n_bins)

    result = mask.copy()
    for b in range(n_bins):
        bin_region = mask & (bin_idx == b)
        if not bin_region.any():
            continue
        footprint = _line_footprint(radius, bin_centers[b])
        # Dilate only THIS bin's own pixels, not the whole mask - otherwise a
        # differently-oriented nearby fibre's pixels would also get smeared
        # using this bin's footprint and could leak into this bin's reach.
        newly_added = dilation(bin_region, footprint=footprint) & ~mask
        if not newly_added.any():
            continue
        result |= newly_added

    return result


# ---------------------------------------------------------------------------
# 4-6. Skeleton endpoints, collinearity-gated geodesic linking, and global
# cost-ranked fragment merging.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    fragment_label: int
    position: tuple[int, int]  # (row, col)
    tangent: tuple[float, float]  # outward-pointing unit vector (row, col)


def _walk_skeleton_branch(skeleton: np.ndarray, start: tuple[int, int], max_steps: int) -> list:
    """Walk up to `max_steps` pixels along a 1px-wide skeleton from `start`,
    stopping at a dead end or a branch point (more than one unvisited
    neighbour). Returns the visited pixels including `start`, in walk order.
    """
    visited = [start]
    seen = {start}
    current = start
    for _ in range(max_steps):
        r, c = current
        neighbors = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
            and 0 <= r + dr < skeleton.shape[0]
            and 0 <= c + dc < skeleton.shape[1]
            and skeleton[r + dr, c + dc]
            and (r + dr, c + dc) not in seen
        ]
        if len(neighbors) != 1:
            break
        current = neighbors[0]
        visited.append(current)
        seen.add(current)
    return visited


def skeleton_endpoints(
    mask: np.ndarray,
    labeled: np.ndarray,
    *,
    tangent_window: int = 10,
) -> list[Endpoint]:
    """Skeletonize each labelled fragment and locate its endpoints (skeleton
    pixels with exactly one skeleton neighbour), each with an outward-pointing
    local tangent estimated from the last `tangent_window` skeleton pixels
    leading to it (the direction the fibre would continue in if extended).

    Isolated single-pixel fragments (no skeleton neighbours anywhere, so no
    tangent can be estimated) are skipped.
    """
    skeleton = skeletonize(mask)
    kernel_sum = np.zeros_like(skeleton, dtype=np.int32)
    padded = np.pad(skeleton.astype(np.int32), 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            kernel_sum += padded[1 + dr:1 + dr + skeleton.shape[0], 1 + dc:1 + dc + skeleton.shape[1]]
    endpoint_mask = skeleton & (kernel_sum == 1)

    endpoints: list[Endpoint] = []
    for r, c in zip(*np.nonzero(endpoint_mask)):
        frag_label = int(labeled[r, c])
        if frag_label == 0:
            continue
        walked = _walk_skeleton_branch(skeleton, (int(r), int(c)), tangent_window)
        if len(walked) < 2:
            continue
        body = np.asarray(walked[1:], dtype=np.float64)
        endpoint_pos = np.asarray(walked[0], dtype=np.float64)
        direction = endpoint_pos - body.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm < EPS:
            continue
        direction = direction / norm
        endpoints.append(Endpoint(frag_label, (int(r), int(c)), (float(direction[0]), float(direction[1]))))

    return endpoints


def _angle_between(u: tuple[float, float], v: tuple[float, float]) -> float:
    """Angle in degrees between two vectors, in [0, 180]."""
    u_arr, v_arr = np.asarray(u), np.asarray(v)
    cos_theta = np.dot(u_arr, v_arr) / (np.linalg.norm(u_arr) * np.linalg.norm(v_arr) + EPS)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def _geodesic_path(
    cost_field: np.ndarray | None,
    a: tuple[int, int],
    b: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], float]:
    """Minimum-cost path between two points. Uses `route_through_array` on a
    local window of `cost_field` when given (path bends toward supporting
    evidence, e.g. residual probability in the gap); a straight line
    otherwise. Returns (path pixels, mean cost along the path).
    """
    if cost_field is None:
        rr, cc = draw_line(a[0], a[1], b[0], b[1])
        return list(zip(rr.tolist(), cc.tolist())), 1.0

    pad = 2
    r0 = max(0, min(a[0], b[0]) - pad)
    c0 = max(0, min(a[1], b[1]) - pad)
    r1 = min(shape[0], max(a[0], b[0]) + pad + 1)
    c1 = min(shape[1], max(a[1], b[1]) + pad + 1)
    window = cost_field[r0:r1, c0:c1]
    a_local = (a[0] - r0, a[1] - c0)
    b_local = (b[0] - r0, b[1] - c0)
    indices, cost = route_through_array(window, a_local, b_local, fully_connected=True, geometric=True)
    path = [(r + r0, c + c0) for r, c in indices]
    mean_cost = cost / max(len(path) - 1, 1)
    return path, float(mean_cost)


def repair_fiber_gaps(
    mask: np.ndarray,
    probability: np.ndarray | None = None,
    *,
    max_gap: int = 15,
    max_angle_deg: float = 30.0,
    min_object_size: int = 4,
    tangent_window: int = 10,
    connector_width: int = 1,
    use_oriented_closing: bool = True,
    closing_radius: int = 4,
    closing_bins: int = 8,
) -> np.ndarray:
    """Repair fragmentation in a binary fibre mask.

    Pipeline: (1) optionally run `oriented_closing` first as a cheap local
    pass that bridges the shortest gaps along each fibre's own orientation;
    (2) skeletonize what remains, find every fragment's endpoints with a
    local tangent; (3) gate every candidate endpoint-pair by gap distance
    (<= `max_gap`) AND collinearity - the angle between each endpoint's
    outward tangent and the vector toward the other endpoint must be
    <= `max_angle_deg` on *both* sides, which is what stops two distinct,
    nearby-but-differently-aimed fibres from being merged into one (a
    counting error); (4) connect every gated pair via a minimum-cost path
    through `1 - probability` when `probability` is given (else a straight
    line); (5) accept candidates globally, cheapest (shortest gap + least
    angular deviation) first, each fragment endpoint usable at most once.

    `probability` (float [0,1], same shape as `mask`) is optional - pass it
    when available (e.g. saved alongside the mask) so connectors prefer
    paths through whatever residual model confidence exists in the gap;
    without it, connectors are straight lines and only the distance/angle
    gates control what gets bridged.
    """
    if mask.dtype != bool:
        mask = mask > 0
    if min_object_size > 0:
        mask = remove_small_objects(mask, max_size=min_object_size - 1)

    repaired = mask.copy()
    if use_oriented_closing and repaired.any():
        repaired = oriented_closing(repaired, radius=closing_radius, n_bins=closing_bins)

    labeled = label(repaired, connectivity=2)
    if labeled.max() <= 1:
        return repaired  # nothing to link, already a single fragment (or empty)

    endpoints = skeleton_endpoints(repaired, labeled, tangent_window=tangent_window)
    if len(endpoints) < 2:
        return repaired

    cost_field = None
    if probability is not None:
        cost_field = np.clip(1.0 - probability.astype(np.float64), EPS, None)

    candidates = []
    for i, ep_a in enumerate(endpoints):
        for ep_b in endpoints[i + 1:]:
            if ep_a.fragment_label == ep_b.fragment_label:
                continue
            dist = float(np.hypot(*(np.asarray(ep_a.position) - np.asarray(ep_b.position))))
            if dist > max_gap:
                continue
            vec_ab = np.asarray(ep_b.position) - np.asarray(ep_a.position)
            vec_ba = -vec_ab
            angle_a = _angle_between(ep_a.tangent, tuple(vec_ab))
            angle_b = _angle_between(ep_b.tangent, tuple(vec_ba))
            if angle_a > max_angle_deg or angle_b > max_angle_deg:
                continue
            path, path_cost = _geodesic_path(cost_field, ep_a.position, ep_b.position, mask.shape)
            total_cost = dist + 2.0 * max(angle_a, angle_b) + 50.0 * path_cost
            candidates.append((total_cost, ep_a, ep_b, path))

    candidates.sort(key=lambda c: c[0])

    used_endpoints: set[tuple[int, tuple[int, int]]] = set()
    for _, ep_a, ep_b, path in candidates:
        key_a = (ep_a.fragment_label, ep_a.position)
        key_b = (ep_b.fragment_label, ep_b.position)
        if key_a in used_endpoints or key_b in used_endpoints:
            continue
        used_endpoints.add(key_a)
        used_endpoints.add(key_b)

        connector = np.zeros_like(repaired)
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        connector[rows, cols] = True
        if connector_width > 1:
            connector = dilation(connector, disk(connector_width - 1))
        repaired |= connector

    return repaired

def _find_prediction(pred_dir: Path, split: str, stem: str, suffix: str) -> Path | None:
    """Mirror evaluate_predictions.py's/postprocess_masks.py's lookup: split
    subfolder first, flat as fallback."""
    candidates = [
        pred_dir / SPLIT_DIRS[split] / f"{stem}{suffix}",
        pred_dir / f"{stem}{suffix}",
    ]
    return next((p for p in candidates if p.exists()), None)


def main():
    parser = argparse.ArgumentParser(
        description="Repair fragmentation (\"holes\") in fibre masks/predictions: oriented "
        "closing plus collinearity-gated skeleton-endpoint linking, so length/aspect-ratio "
        "fibre counting isn't measuring fragments of what should be one whole fibre."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pred-dir", required=True,
        help="Folder of masks to repair, e.g. predict_all.py's --out-dir (or its "
        "postprocessed/ subfolder).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write repaired masks + metrics.csv (default: <pred-dir>/gap_repaired).",
    )
    parser.add_argument("--suffix", default="_pred.tif", help="Prediction filename suffix.")
    parser.add_argument(
        "--max-gap", type=int, default=15,
        help="Maximum gap distance (px) a candidate link may bridge (default: 15).",
    )
    parser.add_argument(
        "--max-angle-deg", type=float, default=30.0,
        help="Maximum angular deviation (degrees) between a fragment's local tangent and the "
        "direction toward the candidate partner, on both sides - the collinearity gate that "
        "stops distinct fibres from being merged (default: 30).",
    )
    parser.add_argument(
        "--min-object-size", type=int, default=4,
        help="Discard connected components smaller than this many pixels before repair, so "
        "isolated noise specks aren't treated as fibre fragments (default: 4).",
    )
    parser.add_argument(
        "--connector-width", type=int, default=1,
        help="Pixel width of drawn connector paths (default: 1).",
    )
    parser.add_argument(
        "--no-oriented-closing", action="store_true",
        help="Skip the oriented-closing pre-pass and go straight to skeleton-endpoint linking.",
    )
    parser.add_argument(
        "--closing-radius", type=int, default=4,
        help="Oriented-closing reach in pixels, if not disabled (default: 4).",
    )
    parser.add_argument(
        "--closing-bins", type=int, default=8,
        help="Number of discrete orientation bins for oriented closing (default: 8).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs = find_pairs(cfg.data)

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir / "gap_repaired"
    split_out_dirs = {split: out_dir / name for split, name in SPLIT_DIRS.items()}
    for d in split_out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for i, pair in enumerate(pairs, start=1):
        pred_path = _find_prediction(pred_dir, pair.split, pair.image_path.stem, args.suffix)
        if pred_path is None:
            missing.append(pair.image_path.name)
            continue

        print(f"[{i}/{len(pairs)}] Repairing {pair.image_path.name} ({pair.split}) ...")
        raw_mask = _read_gray(pred_path)
        repaired = repair_fiber_gaps(
            raw_mask,
            probability=None,
            max_gap=args.max_gap,
            max_angle_deg=args.max_angle_deg,
            min_object_size=args.min_object_size,
            connector_width=args.connector_width,
            use_oriented_closing=not args.no_oriented_closing,
            closing_radius=args.closing_radius,
            closing_bins=args.closing_bins,
        )
        repaired_mask = repaired.astype(np.uint8) * 255

        out_path = split_out_dirs[pair.split] / f"{pair.image_path.stem}{args.suffix}"
        save_mask(repaired_mask, out_path)

        gt_mask = _read_gray(pair.mask_path)
        raw_metrics = compute_metrics(
            raw_mask, gt_mask,
            alpha=cfg.train.loss.tversky_alpha, beta=cfg.train.loss.tversky_beta,
        )
        repaired_metrics = compute_metrics(
            repaired_mask, gt_mask,
            alpha=cfg.train.loss.tversky_alpha, beta=cfg.train.loss.tversky_beta,
        )

        row = {"image": pair.image_path.name, "split": pair.split}
        for m in METRIC_NAMES:
            row[f"raw_{m}"] = raw_metrics[m]
            row[f"repaired_{m}"] = repaired_metrics[m]
            row[f"delta_{m}"] = repaired_metrics[m] - raw_metrics[m]
        rows.append(row)

    if missing:
        print(
            f"Warning: {len(missing)} prediction(s) were not found in {pred_dir} "
            f"(first missing: {missing[0]})."
        )
    if not rows:
        raise FileNotFoundError(
            f"No matching prediction files found in {pred_dir} using suffix {args.suffix!r}."
        )

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    raw_means = {m: statistics.fmean(row[f"raw_{m}"] for row in rows) for m in METRIC_NAMES}
    repaired_means = {m: statistics.fmean(row[f"repaired_{m}"] for row in rows) for m in METRIC_NAMES}
    delta_means = {m: repaired_means[m] - raw_means[m] for m in METRIC_NAMES}

    print(f"Wrote {len(rows)} repaired masks + metrics to {out_dir}")
    print("Raw mean metrics:      " + ", ".join(f"{k}={v:.4f}" for k, v in raw_means.items()))
    print("Repaired mean metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in repaired_means.items()))
    print("Mean delta (repaired - raw): " + ", ".join(f"{k}={v:+.4f}" for k, v in delta_means.items()))


if __name__ == "__main__":
    main()
