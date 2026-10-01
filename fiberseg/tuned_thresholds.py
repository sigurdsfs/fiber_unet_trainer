# tuned_thresholds.py
"""Tuned decision thresholds stored inside a checkpoint.

`fiberseg.tools.auto_tune` writes a dict under `TUNED_KEY` into the best checkpoint
after training; `FiberSegmentationLitModule.on_load_checkpoint` reads it back and
`predict_tiles.load_predictor` calls `apply_tuned` so every prediction uses the
validation-tuned values instead of the config's defaults. Kept dependency-free so
lit_module, predict_tiles and the tools can all import it without cycles.

Stored format (version 1):
    {"version": 1, "split": "val", "metric": "dice",
     "modes": {"fixed": {"threshold": t},
               "hysteresis": {"low": l, "high": h},
               "ridge": {"ridge_threshold": r, "ridge_method": m, "ridge_sigmas": [...]}},
     "val": {mode: stats}, "test": {mode: stats}}
"""
from __future__ import annotations

TUNED_KEY = "fiberseg_tuned_thresholds"


def apply_tuned(cfg, tuned: dict | None) -> list[str]:
    """Write every stored mode's tuned values into `cfg` (in place).

    All tuned modes are applied, so switching `inference.threshold_mode` later still
    gets calibrated values; `threshold_mode` itself is left as configured. Ridge
    thresholds are only meaningful for the filter they were tuned on, so the stored
    ridge method/sigmas are applied with them. Returns a readable list of changes.
    """
    if not tuned:
        return []
    modes = tuned.get("modes", {})
    applied = []
    if "fixed" in modes:
        cfg.train.threshold = float(modes["fixed"]["threshold"])
        applied.append(f"fixed threshold={cfg.train.threshold:.3f}")
    if "hysteresis" in modes:
        cfg.inference.hysteresis_low = float(modes["hysteresis"]["low"])
        cfg.inference.hysteresis_high = float(modes["hysteresis"]["high"])
        applied.append(
            f"hysteresis low={cfg.inference.hysteresis_low:.3f} "
            f"high={cfg.inference.hysteresis_high:.3f}"
        )
    if "ridge" in modes:
        r = modes["ridge"]
        cfg.inference.ridge_threshold = float(r["ridge_threshold"])
        cfg.inference.ridge_method = r.get("ridge_method", cfg.inference.ridge_method)
        cfg.inference.ridge_sigmas = list(r.get("ridge_sigmas", cfg.inference.ridge_sigmas))
        applied.append(
            f"ridge threshold={cfg.inference.ridge_threshold:.3f} "
            f"({cfg.inference.ridge_method})"
        )
    return applied
