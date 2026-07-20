# fiberseg/tools/lr_range_test.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from ..config import AppConfig, load_config
from ..dataset import FiberDataModule
from ..lit_module import FiberSegmentationLitModule
from ..train import resolve_trainer_settings


def run_lr_range_test(
    cfg: AppConfig,
    *,
    min_lr: float,
    max_lr: float,
    num_training: int,
    mode: str,
    early_stop_threshold: float | None,
    out_path: Path,
):
    if cfg.train.matmul_precision:
        torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    datamodule = FiberDataModule(cfg.data, cfg.augmentations)
    datamodule.setup("fit")
    model = FiberSegmentationLitModule(cfg.model, cfg.train)

    accelerator, devices, precision = resolve_trainer_settings(cfg)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        max_epochs=1,
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_training,
        mode=mode,
        early_stop_threshold=early_stop_threshold,
        # Our LightningModule keeps its learning rate nested in `train_cfg`, not a
        # top-level `self.lr`/`self.learning_rate` attribute Lightning can
        # auto-detect, and this model instance is throwaway anyway - the point of
        # this tool is to print/plot a suggestion for the user to put in their
        # config, not to mutate a live model.
        update_attr=False,
    )

    if lr_finder is None:
        raise RuntimeError("LR range test did not run (is fast_dev_run enabled?).")

    suggestion = lr_finder.suggestion()

    fig = lr_finder.plot(suggest=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    print("=" * 80)
    print(f"Tested {len(lr_finder.results['lr'])} learning rates from {min_lr:g} to {max_lr:g} ({mode})")
    print(f"Suggested learning rate: {suggestion!r}")
    print(f"Plot saved to: {out_path.resolve()}")
    print("=" * 80)
    print("Not applied automatically - set train.learning_rate in your config to use it.")

    return suggestion, lr_finder


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a learning-rate range test (Smith LR finder): ramps the LR from "
            "min to max over a short run and plots loss vs. LR, to help pick "
            "train.learning_rate before a full training run."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--max-lr", type=float, default=1.0)
    parser.add_argument("--num-training", type=int, default=100, help="Number of LR steps to try.")
    parser.add_argument("--mode", choices=["exponential", "linear"], default="exponential")
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=4.0,
        help="Stop early once loss exceeds threshold * best loss so far. Set to 0 to disable.",
    )
    parser.add_argument(
        "--out",
        default="lr_range_test.png",
        help="Output path for the loss-vs-lr plot.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    early_stop_threshold = args.early_stop_threshold if args.early_stop_threshold > 0 else None

    run_lr_range_test(
        cfg,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.num_training,
        mode=args.mode,
        early_stop_threshold=early_stop_threshold,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
