# extract_micronet_weights.py
"""Snapshot MicroNet pretrained ENCODER weights to a local file, to unpin smp.

Installing NASA `pretrained-microscopy-models` force-downgrades
segmentation-models-pytorch (0.5.x -> 0.2.1) and timm environment-wide, because
`pmm` pins those old versions. But `pmm`'s only real contribution is a set of
pretrained encoder weights it downloads from a URL - once those weights are saved
locally, `pmm` (and the downgrade) are no longer needed at train time.

This script builds the MicroNet model once (needs `pmm` installed), extracts just
the encoder state_dict, and writes it to a .pth. Afterwards you can:

    pip install "segmentation-models-pytorch>=0.5" "timm>=1.0"   # restore modern smp

and load the saved weights into a normal smp encoder via
`models.create_model` extended to accept a local weights path (see IMPROVEMENTS.md
"Unpin smp" for the exact wiring). This also unlocks modern encoders (ConvNeXt,
SegFormer/MiT) for sweeps.

Run (with pmm still installed):
    python -m fiberseg.tools.extract_micronet_weights --config <micronet-cfg> \
        --out pretrained/micronet_<encoder>.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import load_config
from ..models import create_micronet_model


def main():
    parser = argparse.ArgumentParser(
        description="Save MicroNet pretrained encoder weights locally so smp can be "
        "un-downgraded."
    )
    parser.add_argument("--config", required=True, help="A config with encoder_weights: micronet.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .pth (default: pretrained/micronet_<encoder>.pth).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if str(cfg.model.encoder_weights).lower() != "micronet":
        raise SystemExit(
            f"Config model.encoder_weights is {cfg.model.encoder_weights!r}, expected 'micronet'."
        )

    print(f"Building MicroNet model ({cfg.model.architecture}/{cfg.model.encoder_name}) ...")
    model = create_micronet_model(cfg.model)

    if not hasattr(model, "encoder"):
        raise SystemExit("Created model has no .encoder attribute; cannot extract encoder weights.")

    encoder_state = {k: v.cpu() for k, v in model.encoder.state_dict().items()}

    out = Path(args.out) if args.out else Path("pretrained") / f"micronet_{cfg.model.encoder_name}.pth"  # noqa: E501
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_name": cfg.model.encoder_name,
            "source": "micronet",
            "state_dict": encoder_state,
        },
        out,
    )
    n = sum(v.numel() for v in encoder_state.values())
    print(f"Saved encoder weights ({n:,} params) to {out.resolve()}")
    print(
        "You can now restore modern smp: "
        "pip install 'segmentation-models-pytorch>=0.5' 'timm>=1.0'"
    )


if __name__ == "__main__":
    main()
