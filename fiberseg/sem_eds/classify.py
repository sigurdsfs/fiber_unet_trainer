"""Spectrum evaluation (phase 5).

Two routes:
  * classify(net_counts(...))  - handoff rule set on background-subtracted COUNT ratios.
    Thresholds are illustrative; calibrate on reference asbestos standards on this instrument.
  * ESPRIT quantification      - QuantaxMicroscope.quantify_point(index, cfg.quant_method)
    returns ESPRIT's result text (real at.% / wt.%). parse_quant_result is a stub until the
    text format has been seen on the instrument.

Neither route separates asbestiform fibres from cleavage fragments; TEM/SAED confirms.
"""

from __future__ import annotations

import numpy as np

from .microscope import LINES_KEV
from .settings import SemEdsConfig


def net_counts(
    spec: np.ndarray, calib: tuple[float, float], bg: np.ndarray | None, window_kev: float = 0.08
) -> dict[str, float]:
    off, gain = calib
    e = off + gain * np.arange(len(spec))
    s = spec - bg if bg is not None else spec
    return {
        el: float(max(s[np.abs(e - E) <= window_kev].sum(), 0.0)) for el, E in LINES_KEV.items()
    }


def classify(nc: dict[str, float], cfg: SemEdsConfig) -> tuple[str, dict]:
    if nc["Si"] < cfg.min_si_counts:
        return (
            "non-silicate/other" if sum(nc.values()) > cfg.min_si_counts else "insufficient signal"
        ), {}
    r = {el: nc[el] / nc["Si"] for el in ("Mg", "Fe", "Ca", "Na", "Al")}
    if r["Na"] > 0.10 and r["Fe"] > 0.30:
        cls = "crocidolite"
    elif r["Ca"] > 0.15 and r["Mg"] > 0.20:
        cls = "tremolite/actinolite"
    elif r["Fe"] > 0.40 and r["Ca"] < 0.10 and r["Na"] < 0.10:
        cls = "amosite"
    elif r["Mg"] > 0.90 and r["Fe"] < 0.20 and r["Ca"] < 0.10:
        cls = "chrysotile"
    elif 0.50 < r["Mg"] <= 0.90 and r["Ca"] < 0.10:
        cls = "anthophyllite"
    else:
        cls = "other silicate"
    return cls, r


def parse_quant_result(text: str) -> dict[str, float]:
    """ESPRIT QuantifySpectrum result text -> {element: at.%}.

    TODO(phase 5): the result format is not documented in the headers. Capture a real
    result (bring-up step 5 saves it to quant_example.txt) and implement the parser from it.
    """
    raise NotImplementedError("Implement once a real ESPRIT quant result has been captured")
