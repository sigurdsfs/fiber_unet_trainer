# fiberseg/tools/export_torchscript.py
"""Export a trained Lightning checkpoint to a standalone TorchScript model package.

By default, the exported `model_config.yaml`'s `inference.threshold` is a straight
passthrough of the training config's `train.threshold` - NOT re-optimized. Pass
`--tune-threshold` to sweep it on `--tune-split` first (via `tools.tune_threshold.
find_best_threshold`, same mechanism as `predict_all.py --tune-threshold`) and export
that value instead; a `threshold_pr_curve.png` is written alongside the model, and the
`threshold_tuning` section of model_config.yaml records whether tuning ran and the
stats at the chosen threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import load_config, to_dict
from ..dataset import _hw
from ..lit_module import FiberSegmentationLitModule
from ..tuned_thresholds import apply_tuned
from .tune_threshold import find_best_threshold, plot_pr_curve


def _make_export_config(
    cfg,
    *,
    model_name: str,
    checkpoint_path: str,
    tune_threshold: bool = False,
    tune_split: str | None = None,
    tune_metric: str | None = None,
    tune_steps: int | None = None,
    tune_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    patch_h, patch_w = _hw(cfg.data.patch_size)
    stride_h, stride_w = _hw(cfg.data.stride or cfg.data.patch_size)

    image_channels = int(getattr(cfg.data, "image_channels", cfg.model.in_channels))

    return {
        "format": "torchscript",
        "format_version": 1,
        "model_name": model_name,
        "checkpoint_source": str(checkpoint_path),
        "model": {
            "architecture": cfg.model.architecture,
            "encoder_name": cfg.model.encoder_name,
            "encoder_weights": cfg.model.encoder_weights,
            "in_channels": cfg.model.in_channels,
            "classes": cfg.model.classes,
        },
        "inference": {
            "patch_size": [patch_h, patch_w],
            "stride": [stride_h, stride_w],
            "threshold": float(cfg.train.threshold),
            "image_channels": image_channels,
            "input_channels": image_channels,
            "input_range": [0.0, 1.0],
            "input_shape": "NCHW",
            "output": "logits",
            "apply_sigmoid": True,
            "mask_foreground_value": 255,
        },
        "normalization": {
            "kind": "percentile_per_image",
            "percentiles": [1.0, 99.5],
            "description": (
                "Match training preprocessing. For the full source image (before "
                "tiling): convert to float32, replace NaN/inf, normalize using the "
                "1.0 and 99.5 percentiles, then clip to [0, 1]. Tiles must be cropped "
                "from this already-normalized image, not normalized individually."
            ),
        },
        "threshold_tuning": {
            "tuned": bool(tune_threshold),
            "split": tune_split if tune_threshold else None,
            "metric": tune_metric if tune_threshold else None,
            "steps": tune_steps if tune_threshold else None,
            "stats_at_best": tune_stats if tune_threshold else None,
        },
        "training_config_snapshot": to_dict(cfg),
    }


def export_torchscript(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    device_name: str = "cpu",
    verify: bool = True,
    tune_threshold: bool = False,
    tune_split: str = "val",
    tune_metric: str = "dice",
    tune_steps: int = 99,
) -> None:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    config_out_path = out_dir / "model_config.yaml"

    cfg = load_config(config_path)
    device = torch.device(device_name)

    lit_model = FiberSegmentationLitModule.load_from_checkpoint(
        str(checkpoint_path),
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        map_location=device,
    )
    applied = apply_tuned(cfg, lit_model.tuned_thresholds)
    if applied:
        print("Using validation-tuned thresholds from checkpoint: " + "; ".join(applied))

    lit_model.eval()
    lit_model.to(device)

    tune_stats = None
    if tune_threshold:
        best_threshold, tune_stats, tune_thresholds, tune_metrics = find_best_threshold(
            cfg, lit_model, device,
            split=tune_split, metric=tune_metric, steps=tune_steps,
        )
        print(
            f"Tuned train.threshold on the {tune_split} split: {best_threshold:.3f} "
            f"({tune_metric}={tune_stats[tune_metric]:.4f})."
        )
        cfg.train.threshold = best_threshold

        pr_curve_path = out_dir / "threshold_pr_curve.png"
        pr_auc = plot_pr_curve(
            tune_thresholds, tune_metrics, best_threshold, pr_curve_path,
            title=f"Precision-recall curve, split={tune_split!r}",
        )
        print(f"Wrote precision-recall curve (AUC-PR={pr_auc:.4f}) to {pr_curve_path}")

    model = lit_model.model
    model.eval()
    model.to(device)

    patch_h, patch_w = _hw(cfg.data.patch_size)
    example = torch.zeros(
        1,
        int(cfg.model.in_channels),
        patch_h,
        patch_w,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)

    traced.save(str(model_path))

    export_config = _make_export_config(
        cfg,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        tune_threshold=tune_threshold,
        tune_split=tune_split,
        tune_metric=tune_metric,
        tune_steps=tune_steps,
        tune_stats=tune_stats,
    )

    with open(config_out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(export_config, f, sort_keys=False)

    if verify:
        loaded = torch.jit.load(str(model_path), map_location=device)
        loaded.eval()

        with torch.no_grad():
            original_out = model(example)
            exported_out = loaded(example)

        max_abs_diff = float((original_out - exported_out).abs().max().cpu())
        print(f"Verification max abs diff: {max_abs_diff:.8f}")

        if max_abs_diff > 1e-2:
            raise RuntimeError(
                f"Export verification failed. Difference was too large: {max_abs_diff}"
            )

    print("Export complete.")
    print(f"Model:  {model_path.resolve()}")
    print(f"Config: {config_out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained Lightning checkpoint to a standalone TorchScript model package."
        )
    )
    parser.add_argument("--config", required=True, help="Training YAML config used for the model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .ckpt checkpoint.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for model.pt and model_config.yaml.",
    )
    parser.add_argument(
        "--model-name",
        default="fiber_unet",
        help="Human-readable exported model name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used during export. CPU is safest for deployment exports.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip loading the exported model and comparing outputs.",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Re-tune train.threshold on --tune-split before exporting, so the exported "
        "model_config.yaml records the tuned value instead of the training config's as-is "
        "train.threshold.",
    )
    parser.add_argument(
        "--tune-split",
        default="val",
        choices=["train", "val", "test"],
        help="Split to tune train.threshold on when --tune-threshold is set (default: val; "
        "never tune on test for reporting).",
    )
    parser.add_argument(
        "--tune-metric",
        default="dice",
        choices=["dice", "iou", "f2", "tversky"],
        help="Metric to maximize when --tune-threshold is set (default: dice).",
    )
    parser.add_argument(
        "--tune-steps",
        type=int,
        default=99,
        help="Thresholds tried in (0,1) when --tune-threshold is set (default: 99).",
    )

    args = parser.parse_args()

    export_torchscript(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        model_name=args.model_name,
        device_name=args.device,
        verify=not args.no_verify,
        tune_threshold=args.tune_threshold,
        tune_split=args.tune_split,
        tune_metric=args.tune_metric,
        tune_steps=args.tune_steps,
    )


if __name__ == "__main__":
    main()