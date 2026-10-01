"""Run settings for the SEM tiling -> segmentation -> EDS loop.

Defaults live in `SemEdsConfig`. Instrument-specific values (DLL path, server, login, stage
limits) are loaded from a local YAML (`configs/sem_eds_local.yaml`, gitignored; copy
`configs/sem_eds_local.example.yaml`). Unlike fiberseg.config, unknown keys are an ERROR
here - a silently ignored stage limit is not acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class SemEdsConfig:
    # --- Connection (from local YAML) -------------------------------------------------
    dll_path: str = r"C:\Program Files\Bruker\Bruker.API.Esprit64.dll"  # TODO: installed path
    server: str = "Lokaler Server"  # check with bringup step1 (QueryServers)
    user: str = ""
    password: str = ""
    tcp_host: str = ""  # empty -> OpenClient (local); else OpenClientTCP
    tcp_port: int = 5328
    spu: int = 1  # spectrometer index

    # --- Tiling (stage units in the config are um; the API wants mm) ------------------
    grid_cols: int = 10
    grid_rows: int = 5
    origin_um: tuple = (0.0, 0.0)  # stage (x, y) of the first tile centre
    image_px: tuple = (1024, 768)  # (width, height)
    pixel_size_um: float = 0.05  # acquisition pixel size; asserted per image
    pixel_size_tol: float = 0.02  # relative tolerance for that assertion
    overlap: float = 0.10
    stage_settle_s: float = 1.0
    stage_timeout_s: float = 30.0
    image_average: int = 16  # ImageSetConfiguration 'Average' (dwell cycles)
    # Hard stage box (um). Moves outside it are refused, in addition to GetSEMStageRange.
    stage_x_limits_um: tuple = (0.0, 0.0)
    stage_y_limits_um: tuple = (0.0, 0.0)
    # Stage axes vs image axes: TODO(bring-up step 6) - measure sign/rotation.
    stage_x_sign: int = 1
    stage_y_sign: int = 1

    # --- Fibre counting criteria (ISO 14966 / VDI 3492) -------------------------------
    min_length_um: float = 5.0
    max_width_um: float = 3.0
    min_aspect: float = 3.0

    # --- Targeting ---------------------------------------------------------------------
    border_margin_px: int = 10
    crossing_margin_px: int = 4
    branch_margin_px: int = 3
    points_per_fiber: int = 2
    min_point_spacing_um: float = 1.0
    bg_offset_um: float = 1.0
    max_fibers_per_tile: int = 30

    # --- EDS ---------------------------------------------------------------------------
    eds_real_time_ms: int = 5000  # StartPointListMeasurement takes REAL time (ms)
    point_list_index_base: int = 1  # TODO(bring-up step 4): docs say 1..n, samples use 0..n-1
    point_list_wait: str = (
        "poll_state"  # TODO(bring-up step 4): "blocking" | "poll_state" | "sleep"
    )
    spectrum_buf_bytes: int = 65536
    min_si_counts: float = 200.0
    quant_method: str = ""  # e.g. "default.mtdx"; empty -> count-ratio classify()

    # --- Segmentation ------------------------------------------------------------------
    seg_config: str = "configs/example.yaml"  # fiberseg training config of the checkpoint
    ckpt: str = ""
    # Pixel size the checkpoint was trained at. Models trained on data resampled by
    # tools.standardize_pixel_size have one (e.g. the *_pixelstd_50nm configs -> 0.05);
    # models trained on the raw mixed-scale data (~14-90 nm/px) have none -> leave 0 (no
    # resampling). Not stored in the training config, so set it per checkpoint.
    training_pixel_size_um: float = 0.0
    min_object_px: int = 20

    # --- QC ----------------------------------------------------------------------------
    drift_check: bool = True
    seed: int = 0

    @property
    def step_um(self) -> tuple[float, float]:
        w, h = self.image_px
        s = self.pixel_size_um * (1.0 - self.overlap)
        return (w * s, h * s)

    def tile_positions(self) -> list[tuple[int, int, float, float]]:
        """Serpentine raster -> (row, col, x_um, y_um). Minimises stage travel."""
        ox, oy = self.origin_um
        sx, sy = self.step_um
        out = []
        for r in range(self.grid_rows):
            cols = range(self.grid_cols) if r % 2 == 0 else reversed(range(self.grid_cols))
            for c in cols:
                out.append((r, c, ox + self.stage_x_sign * c * sx, oy + self.stage_y_sign * r * sy))
        return out

    def public_dict(self) -> dict:
        """Everything except credentials - this is what gets written to config.json."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("user", "password")
        }


def load_settings(path: str | Path | None, **overrides) -> SemEdsConfig:
    values: dict = {}
    if path:
        values.update(yaml.safe_load(Path(path).read_text()) or {})
    values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(SemEdsConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown sem_eds setting(s): {unknown}")
    for k, v in values.items():
        if isinstance(v, list):
            values[k] = tuple(v)
    return SemEdsConfig(**values)
