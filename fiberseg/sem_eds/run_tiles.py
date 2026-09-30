"""Closed-loop SEM tiling -> segmentation -> targeted point-list EDS (phase 4).

For each stage tile: move -> acquire image -> segment -> pick EDS points on countable fibres
(+ one background point each) -> point-list EDS on the same frame -> classify -> save.

    python -m fiberseg.sem_eds.run_tiles --dry-run --out runs/test
    python -m fiberseg.sem_eds.run_tiles --dry-run --ckpt best.ckpt --seg-config configs/x.yaml
    python -m fiberseg.sem_eds.run_tiles --settings configs/sem_eds_local.yaml --tiles 2x2

Output: <out>/config.json, fibers.csv, tile_XXX/{image.tif, labels.npy, spectra.npy, points.json}
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation

from .classify import classify, net_counts
from .microscope import LINES_KEV, Microscope, MockMicroscope, QuantaxMicroscope
from .segmenter import FiberSegSegmenter, ThresholdSegmenter
from .settings import SemEdsConfig, load_settings
from .targeting import build_point_list, find_targets

CSV_FIELDS = [
    "tile",
    "row",
    "col",
    "stage_x_um",
    "stage_y_um",
    "fiber",
    "length_um",
    "width_um",
    "aspect",
    "touches_border",
    "countable",
    "n_points",
    "class",
    "Mg/Si",
    "Fe/Si",
    "Ca/Si",
    "Na/Si",
    "Al/Si",
    "points_rc",
]


def run(cfg: SemEdsConfig, scope: Microscope, segmenter, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.public_dict(), indent=2))
    rng = np.random.default_rng(cfg.seed)
    tiles = cfg.tile_positions()

    with open(out / "fibers.csv", "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        scope.connect()
        try:
            scope.setup(cfg)
            t_start = time.time()
            for ti, (r, c, x, y) in enumerate(tiles):
                tdir = out / f"tile_{ti:03d}"
                tdir.mkdir(exist_ok=True)

                scope.move_stage(x, y)
                time.sleep(cfg.stage_settle_s)
                img = scope.acquire_image()
                tifffile.imwrite(tdir / "image.tif", img)

                labels = segmenter(img)
                np.save(tdir / "labels.npy", labels)
                fibers = find_targets(labels, cfg, rng)
                pts_rc, owner = build_point_list(fibers)

                spectra, calib = None, None
                if pts_rc:
                    pts_xy = np.array(
                        [(cc, rr) for rr, cc in pts_rc], dtype=np.int32
                    )  # x=col, y=row
                    spectra, calib = scope.acquire_point_spectra(pts_xy)
                    np.save(tdir / "spectra.npy", spectra)
                    (tdir / "points.json").write_text(
                        json.dumps({"points_rc": pts_rc, "owner": owner, "calib": calib})
                    )

                drift = None
                if cfg.drift_check and pts_rc:
                    shift, _, _ = phase_cross_correlation(
                        img, scope.acquire_image(), upsample_factor=4
                    )
                    drift = float(np.hypot(*shift) * cfg.pixel_size_um)

                for f in fibers:
                    writer.writerow(_fiber_row(f, ti, r, c, x, y, owner, spectra, calib, cfg))
                csv_f.flush()

                n_cnt = sum(f.countable for f in fibers)
                n_eds = sum(bool(f.targets_rc) for f in fibers)
                eta = (time.time() - t_start) / (ti + 1) * (len(tiles) - ti - 1)
                msg = (
                    f"tile {ti + 1:>3}/{len(tiles)}  objects={len(fibers):>3}"
                    f"  countable={n_cnt:>3}  EDS={n_eds:>3}"
                )
                if drift is not None:
                    msg += f"  drift={drift:.3f} um"
                print(msg + f"  ETA {eta / 60:.1f} min")
        finally:
            scope.close()


def _fiber_row(f, ti, r, c, x, y, owner, spectra, calib, cfg) -> dict:
    row = dict(
        tile=ti,
        row=r,
        col=c,
        stage_x_um=round(x, 2),
        stage_y_um=round(y, 2),
        fiber=f.label,
        length_um=round(f.length_um, 2),
        width_um=round(f.width_um, 3),
        aspect=round(f.aspect, 1),
        touches_border=f.touches_border,
        countable=f.countable,
        n_points=len(f.targets_rc),
        points_rc=json.dumps(f.targets_rc),
    )
    if f.targets_rc and spectra is not None:
        idx = [i for i, o in enumerate(owner) if o == (f.label, "fiber")]
        bidx = [i for i, o in enumerate(owner) if o == (f.label, "bg")]
        bg = spectra[bidx[0]] if bidx else None
        nc = [net_counts(spectra[i], calib, bg) for i in idx]
        mean_nc = {el: float(np.mean([d[el] for d in nc])) for el in LINES_KEV}
        # TODO(phase 5): if cfg.quant_method, use scope.quantify_point(i, cfg.quant_method)
        # + classify.parse_quant_result instead of count ratios.
        cls, ratios = classify(mean_nc, cfg)
        row["class"] = cls
        for el, v in ratios.items():
            row[f"{el}/Si"] = round(v, 3)
    elif f.countable:
        row["class"] = "not analysed"
    return row


def confirm_real_run(cfg: SemEdsConfig) -> None:
    tiles = cfg.tile_positions()
    xs, ys = [t[2] for t in tiles], [t[3] for t in tiles]
    print(
        f"REAL INSTRUMENT RUN: {len(tiles)} tiles, {cfg.image_px[0]}x{cfg.image_px[1]} px at "
        f"{cfg.pixel_size_um} um/px"
    )
    print(f"  stage x {min(xs):.1f}..{max(xs):.1f} um (limits {cfg.stage_x_limits_um})")
    print(f"  stage y {min(ys):.1f}..{max(ys):.1f} um (limits {cfg.stage_y_limits_um})")
    print(f"  EDS real time {cfg.eds_real_time_ms} ms/point")
    if input("Type 'yes' to start: ").strip().lower() != "yes":
        raise SystemExit("aborted")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--settings", help="local YAML with instrument settings (sem_eds_local.yaml)")
    ap.add_argument("--ckpt", help="Lightning checkpoint (.ckpt)")
    ap.add_argument("--seg-config", help="fiberseg training config that belongs to the checkpoint")
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--dry-run", action="store_true", help="simulated microscope")
    ap.add_argument("--pixel-size", type=float, help="override acquisition pixel size (um)")
    ap.add_argument("--tiles", help="override grid as COLSxROWS, e.g. 10x5")
    args = ap.parse_args()

    grid = (
        dict(zip(("grid_cols", "grid_rows"), map(int, args.tiles.lower().split("x"))))
        if args.tiles
        else {}
    )
    cfg = load_settings(
        args.settings,
        ckpt=args.ckpt,
        seg_config=args.seg_config,
        pixel_size_um=args.pixel_size,
        **grid,
    )

    segmenter = FiberSegSegmenter(cfg) if cfg.ckpt else ThresholdSegmenter(cfg)
    if args.dry_run:
        cfg.stage_settle_s = 0.0
        scope: Microscope = MockMicroscope(cfg.seed)
    else:
        if not cfg.ckpt:
            ap.error("--ckpt (or ckpt in --settings) is required unless --dry-run")
        confirm_real_run(cfg)
        scope = QuantaxMicroscope(cfg)

    run(cfg, scope, segmenter, Path(args.out))


if __name__ == "__main__":
    main()
