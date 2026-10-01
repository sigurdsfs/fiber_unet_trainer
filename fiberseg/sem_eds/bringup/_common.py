"""Shared helpers for the bring-up scripts. Run them in order, on the instrument PC, with the
operator present:

    python -m fiberseg.sem_eds.bringup.stepN_xxx --settings configs/sem_eds_local.yaml
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from ..microscope import QuantaxMicroscope
from ..settings import SemEdsConfig, load_settings


def parse(description: str, extra=None) -> tuple[argparse.Namespace, SemEdsConfig]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--settings", default="configs/sem_eds_local.yaml")
    ap.add_argument("--out", default="runs/bringup")
    if extra:
        extra(ap)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return args, load_settings(args.settings)


def confirm(what: str) -> None:
    if input(f"{what}\nType 'yes' to continue: ").strip().lower() != "yes":
        raise SystemExit("aborted")


@contextmanager
def connected(cfg: SemEdsConfig):
    scope = QuantaxMicroscope(cfg)
    scope.connect()
    print(f"connected, CID={scope.cid.value}")
    try:
        yield scope
    finally:
        scope.close()
        print("connection closed (ESPRIT still running)")
