"""Fibre measurement (ISO 14966 / VDI 3492 criteria) and EDS target selection.

Logic unchanged from the handoff's tile_segment_eds.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

from .settings import SemEdsConfig


@dataclass
class Fiber:
    label: int
    length_um: float
    width_um: float
    aspect: float
    touches_border: bool
    countable: bool
    targets_rc: list = field(default_factory=list)  # [(row, col), ...]
    bg_rc: tuple | None = None


def _branch_points(skel: np.ndarray) -> np.ndarray:
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    return skel & (nb > 2)


def find_targets(labels: np.ndarray, cfg: SemEdsConfig, rng: np.random.Generator) -> list[Fiber]:
    px = cfg.pixel_size_um
    H, W = labels.shape
    m = cfg.border_margin_px
    occupied = ndi.binary_dilation(labels > 0, structure=morphology.disk(cfg.crossing_margin_px))
    fibers = []

    for rp in measure.regionprops(labels):
        r0, c0, r1, c1 = rp.bbox
        skel = morphology.skeletonize(rp.image)
        n_skel = max(int(skel.sum()), 1)
        length_um = max(rp.feret_diameter_max, n_skel) * px
        width_um = rp.area / n_skel * px
        aspect = length_um / max(width_um, 1e-9)
        touches = r0 < m or c0 < m or r1 > H - m or c1 > W - m
        countable = (
            length_um >= cfg.min_length_um
            and width_um < cfg.max_width_um
            and aspect >= cfg.min_aspect
        )
        f = Fiber(rp.label, length_um, width_um, aspect, touches, countable)
        fibers.append(f)
        if not countable:
            continue

        # Candidate points: skeleton, away from branches, tile edges and other fibres
        branch_zone = ndi.binary_dilation(
            _branch_points(skel), structure=morphology.disk(cfg.branch_margin_px)
        )
        rr, cc = np.nonzero(skel & ~branch_zone)
        rr, cc = rr + r0, cc + c0
        keep = (rr >= m) & (rr < H - m) & (cc >= m) & (cc < W - m)
        rr, cc = rr[keep], cc[keep]
        if len(rr):
            other = occupied & ~ndi.binary_dilation(
                labels == rp.label, structure=morphology.disk(cfg.crossing_margin_px)
            )
            ok = ~other[rr, cc]
            rr, cc = rr[ok], cc[ok]
        if not len(rr):
            continue

        # Prefer the thickest parts of the fibre (most signal, least substrate)
        dist = ndi.distance_transform_edt(labels == rp.label)
        min_sep = cfg.min_point_spacing_um / px
        for i in np.argsort(-dist[rr, cc]):
            p = np.array([rr[i], cc[i]])
            if all(np.hypot(*(p - q)) >= min_sep for q in f.targets_rc):
                f.targets_rc.append((int(p[0]), int(p[1])))
            if len(f.targets_rc) >= cfg.points_per_fiber:
                break

        # Background point perpendicular to the fibre axis
        o = rp.orientation
        perp = np.array([-np.sin(o), np.cos(o)])
        base = np.array(f.targets_rc[0], float)
        for sign in (1, -1):
            q = np.round(base + sign * perp * cfg.bg_offset_um / px).astype(int)
            if m <= q[0] < H - m and m <= q[1] < W - m and not occupied[q[0], q[1]]:
                f.bg_rc = (int(q[0]), int(q[1]))
                break

    targeted = [f for f in fibers if f.targets_rc]
    if len(targeted) > cfg.max_fibers_per_tile:
        keep_ids = set(
            rng.choice([f.label for f in targeted], cfg.max_fibers_per_tile, replace=False)
        )
        for f in targeted:
            if f.label not in keep_ids:
                f.targets_rc, f.bg_rc = [], None
    return fibers


def build_point_list(fibers: list[Fiber]) -> tuple[list[tuple[int, int]], list[tuple[int, str]]]:
    """One point list per tile: each fibre's targets followed by its background point.
    Returns (points_rc, owner) where owner[i] = (fibre label, "fiber" | "bg")."""
    pts_rc, owner = [], []
    for f in fibers:
        for p in f.targets_rc:
            pts_rc.append(p)
            owner.append((f.label, "fiber"))
        if f.targets_rc and f.bg_rc:
            pts_rc.append(f.bg_rc)
            owner.append((f.label, "bg"))
    return pts_rc, owner
