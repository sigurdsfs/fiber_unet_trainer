# callbacks.py
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import MLFlowLogger


class PeriodicPredictionImageLogger(pl.Callback):
    """
    Logs prediction images periodically during training.

    It logs:
      - validation samples only
      - only samples containing fibers
      - every N epochs
      - max_images PNG files each time
    """

    def __init__(
        self,
        max_images: int = 10,
        threshold: float = 0.5,
        artifact_dir: str = "predictions/periodic",
        every_n_epochs: int = 15,
        max_candidate_samples: int = 1000,
    ):
        super().__init__()
        self.max_images = int(max_images)
        self.threshold = float(threshold)
        self.artifact_dir = artifact_dir
        self.every_n_epochs = int(every_n_epochs)
        self.max_candidate_samples = int(max_candidate_samples)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = int(trainer.current_epoch)

        if epoch == 0:
            should_log = True
        else:
            should_log = (epoch + 1) % self.every_n_epochs == 0

        if not should_log:
            return

        logger = trainer.logger

        if not isinstance(logger, MLFlowLogger):
            return

        datamodule = trainer.datamodule

        if datamodule is None:
            return

        val_dataset = getattr(datamodule, "val_ds", None)

        if val_dataset is None or len(val_dataset) == 0:
            return

        metric_value = trainer.callback_metrics.get("val/tversky")
        metric_text = "none"

        if metric_value is not None:
            metric_text = f"{float(metric_value.detach().cpu().item()):.5f}"

        artifact_path = f"{self.artifact_dir}/epoch_{epoch + 1:03d}_val_tversky_{metric_text}"

        _log_prediction_samples(
            logger=logger,
            model=pl_module,
            dataset=val_dataset,
            device=pl_module.device,
            threshold=self.threshold,
            max_images=self.max_images,
            max_candidate_samples=self.max_candidate_samples,
            artifact_path=artifact_path,
            title_prefix=f"Periodic validation epoch={epoch + 1}",
        )


class BestModelPredictionImageLogger(pl.Callback):
    """
    Logs prediction images from the best checkpoint after training.

    It logs:
      - validation samples only
      - only samples containing fibers
      - after training is complete
      - from the best checkpoint according to ModelCheckpoint
      - max_images PNG files
    """

    def __init__(
        self,
        max_images: int = 20,
        threshold: float = 0.5,
        artifact_dir: str = "predictions/best_model",
        max_candidate_samples: int = 1500,
    ):
        super().__init__()
        self.max_images = int(max_images)
        self.threshold = float(threshold)
        self.artifact_dir = artifact_dir
        self.max_candidate_samples = int(max_candidate_samples)

    def on_train_end(self, trainer, pl_module):
        if trainer.fast_dev_run:
            return

        logger = trainer.logger

        if not isinstance(logger, MLFlowLogger):
            return

        datamodule = trainer.datamodule

        if datamodule is None:
            return

        val_dataset = getattr(datamodule, "val_ds", None)

        if val_dataset is None or len(val_dataset) == 0:
            return

        best_checkpoint_path = self._get_best_checkpoint_path(trainer)

        original_device = pl_module.device
        device = original_device

        # Free GPU memory before loading a second copy of the model.
        # This avoids possible out-of-memory errors on the 12 GB RTX A2000.
        pl_module.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if best_checkpoint_path is None:
            model_for_prediction = pl_module
            checkpoint_label = "current_model_no_best_checkpoint_found"
            model_for_prediction.to(device)
        else:
            model_for_prediction = type(pl_module).load_from_checkpoint(
                best_checkpoint_path,
                model_cfg=pl_module.model_cfg,
                train_cfg=pl_module.train_cfg,
                map_location="cpu",
            )
            checkpoint_label = Path(best_checkpoint_path).stem
            model_for_prediction.to(device)

        model_for_prediction.eval()

        artifact_path = f"{self.artifact_dir}/{checkpoint_label}"

        _log_prediction_samples(
            logger=logger,
            model=model_for_prediction,
            dataset=val_dataset,
            device=device,
            threshold=self.threshold,
            max_images=self.max_images,
            max_candidate_samples=self.max_candidate_samples,
            artifact_path=artifact_path,
            title_prefix="Best checkpoint validation",
        )

        # Leave the original LightningModule on CPU after training.
        # trainer.test(..., ckpt_path='best') will load/place the best checkpoint itself.

    def _get_best_checkpoint_path(self, trainer) -> str | None:
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                best_model_path = getattr(callback, "best_model_path", None)

                if best_model_path:
                    return best_model_path

        return None


def _log_prediction_samples(
    logger: MLFlowLogger,
    model,
    dataset,
    device: torch.device,
    threshold: float,
    max_images: int,
    max_candidate_samples: int,
    artifact_path: str,
    title_prefix: str,
):
    selected_indices = _select_indices_with_fibers(
        dataset=dataset,
        max_images=max_images,
        max_candidate_samples=max_candidate_samples,
    )

    if not selected_indices:
        return

    was_training = model.training
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        summary_lines = [
            f"threshold: {threshold}",
            f"max_images: {max_images}",
            f"selected_indices: {selected_indices}",
        ]

        summary_path = tmpdir / "summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        logger.experiment.log_artifact(
            run_id=logger.run_id,
            local_path=str(summary_path),
            artifact_path=artifact_path,
        )

        for image_number, idx in enumerate(selected_indices):
            sample = dataset[idx]

            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                continue

            img, mask = sample[0], sample[1]
            img_batch = img.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_batch)
                prob = torch.sigmoid(logits)[0, 0].detach().cpu()
                pred = (prob > threshold).float()

            fig = _make_figure(
                img=img.cpu(),
                mask=mask.cpu(),
                prob=prob,
                pred=pred,
                title=f"{title_prefix} sample idx={idx}",
            )

            out = tmpdir / f"sample_{image_number:03d}_dataset_idx_{idx:05d}.png"
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)

            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path=str(out),
                artifact_path=artifact_path,
            )

    if was_training:
        model.train()


def _select_indices_with_fibers(
    dataset,
    max_images: int,
    max_candidate_samples: int,
) -> list[int]:
    selected: list[int] = []

    n = len(dataset)

    if n == 0:
        return selected

    n_candidates = min(n, max_candidate_samples)

    # Spread candidates across the validation set instead of only checking the first tiles.
    candidate_indices = np.linspace(
        0,
        n - 1,
        num=n_candidates,
        dtype=int,
    ).tolist()

    seen = set()

    for idx in candidate_indices:
        if idx in seen:
            continue

        seen.add(idx)

        sample = dataset[idx]

        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            continue

        mask = sample[1]
        foreground_pixels = int((mask > 0.5).sum().item())

        if foreground_pixels > 0:
            selected.append(idx)

        if len(selected) >= max_images:
            break

    return selected


def _make_figure(
    img: torch.Tensor,
    mask: torch.Tensor,
    prob: torch.Tensor,
    pred: torch.Tensor,
    title: str,
):
    img_np = _image_to_numpy(img)
    mask_np = _mask_to_numpy(mask)
    prob_np = prob.squeeze().numpy()
    pred_np = pred.squeeze().numpy()

    pred_overlay_np = _make_overlay(
        img_np=img_np,
        mask_np=pred_np,
        color="red",
    )

    gt_overlay_np = _make_overlay(
        img_np=img_np,
        mask_np=mask_np,
        color="green",
    )

    combined_overlay_np = _make_combined_overlay(
        img_np=img_np,
        gt_np=mask_np,
        pred_np=pred_np,
    )

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
    axes[0].set_title("Input")

    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    axes[2].imshow(prob_np, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Probability")

    axes[3].imshow(pred_overlay_np)
    axes[3].set_title("Prediction overlay")

    axes[4].imshow(gt_overlay_np)
    axes[4].set_title("GT overlay")

    axes[5].imshow(combined_overlay_np)
    axes[5].set_title("GT green / Pred red")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


def _image_to_numpy(img: torch.Tensor):
    img = img.detach().cpu().float()

    if img.ndim == 3 and img.shape[0] == 1:
        arr = img[0].numpy()
    elif img.ndim == 3 and img.shape[0] >= 3:
        arr = img[:3].permute(1, 2, 0).numpy()
    elif img.ndim == 2:
        arr = img.numpy()
    else:
        arr = img.squeeze().numpy()

    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)

    return arr


def _mask_to_numpy(mask: torch.Tensor):
    mask = mask.detach().cpu().float().squeeze().numpy()
    return (mask > 0.5).astype(np.float32)


def _make_overlay(img_np: np.ndarray, mask_np: np.ndarray, color: str):
    mask_bool = mask_np > 0.5

    if img_np.ndim == 2:
        base = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        base = img_np.copy()

    overlay = base.copy()

    if color == "red":
        overlay[mask_bool, 0] = 1.0
        overlay[mask_bool, 1] *= 0.25
        overlay[mask_bool, 2] *= 0.25
    elif color == "green":
        overlay[mask_bool, 0] *= 0.25
        overlay[mask_bool, 1] = 1.0
        overlay[mask_bool, 2] *= 0.25
    else:
        raise ValueError(f"Unknown color: {color}")

    return np.clip(overlay, 0.0, 1.0)


def _make_combined_overlay(
    img_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
):
    gt_bool = gt_np > 0.5
    pred_bool = pred_np > 0.5

    if img_np.ndim == 2:
        base = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        base = img_np.copy()

    overlay = base.copy()

    # Ground truth in green.
    overlay[gt_bool, 0] *= 0.25
    overlay[gt_bool, 1] = 1.0
    overlay[gt_bool, 2] *= 0.25

    # Prediction in red.
    overlay[pred_bool, 0] = 1.0
    overlay[pred_bool, 1] *= 0.25
    overlay[pred_bool, 2] *= 0.25

    # True positive overlap in yellow.
    overlap = gt_bool & pred_bool
    overlay[overlap, 0] = 1.0
    overlay[overlap, 1] = 1.0
    overlay[overlap, 2] *= 0.15

    return np.clip(overlay, 0.0, 1.0)