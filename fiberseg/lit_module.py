# lit_module.py

from __future__ import annotations

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig, to_dict
from .models import create_model


def _confusion_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> tuple[float, float, float]:
    """Hard-thresholded tp/fp/fn pixel counts for one batch, as plain floats so callers can
    accumulate them across an epoch without holding onto device tensors."""
    pred = torch.sigmoid(logits) > threshold
    targ = target > 0.5
    tp = (pred & targ).sum().float().item()
    fp = (pred & ~targ).sum().float().item()
    fn = (~pred & targ).sum().float().item()
    return tp, fp, fn


def _stats_from_counts(
    tp: float,
    fp: float,
    fn: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Dice/iou/precision/recall/tversky/f2 from tp/fp/fn counts. Passing counts accumulated
    over a whole epoch (rather than per-batch) avoids empty-tile batches trivially scoring ~0
    and skewing a naive per-batch average on sparse fiber masks."""
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    tversky = tp / (tp + alpha * fp + beta * fn + eps)
    f2 = (5 * precision * recall) / (4 * precision + recall + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tversky": tversky,
        "f2": f2,
    }


def _soft_tversky_index(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-7,
):
    # Differentiable Tversky index used for the loss.
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def _soft_skeletonize(prob: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable morphological skeleton of a soft mask (Shit et al., clDice).

    Repeatedly erodes (min-pool, implemented as -maxpool(-x)) and re-dilates, and
    accumulates the pixels removed by each erosion step - the medial axis. Fully
    differentiable (only min/max pooling), so it can sit inside the loss.
    """
    def _erode(x):
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def _dilate(x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    skel = F.relu(prob - _dilate(_erode(prob)))
    for _ in range(iters):
        prob = _erode(prob)
        delta = F.relu(prob - _dilate(_erode(prob)))
        # skel + delta - skel*delta is a soft (probabilistic) union.
        skel = skel + delta - skel * delta
    return skel


def _soft_cldice(logits: torch.Tensor, target: torch.Tensor, iters: int, eps: float = 1e-7):
    """Soft centerline Dice between predicted and target skeletons.

    Topology-preserving: high only when the predicted skeleton lies inside the
    target mask AND the target skeleton lies inside the prediction, so it directly
    rewards keeping thin fibers connected rather than merely overlapping in area.
    """
    probs = torch.sigmoid(logits)
    target = target.float()

    skel_pred = _soft_skeletonize(probs, iters)
    skel_true = _soft_skeletonize(target, iters)

    dims = tuple(range(1, probs.ndim))
    # tprec: predicted skeleton covered by the true mask; tsens: true skeleton
    # covered by the predicted mask.
    tprec = (skel_pred * target).sum(dim=dims) + eps
    tprec = tprec / (skel_pred.sum(dim=dims) + eps)
    tsens = (skel_true * probs).sum(dim=dims) + eps
    tsens = tsens / (skel_true.sum(dim=dims) + eps)
    return 2.0 * tprec * tsens / (tprec + tsens)


class FiberSegmentationLitModule(pl.LightningModule):
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.save_hyperparameters({"model": to_dict(model_cfg), "train": to_dict(train_cfg)})
        self.model = create_model(model_cfg)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, x):
        return self.model(x)

    def _tversky_loss(self, logits, mask):
        ti = _soft_tversky_index(
            logits,
            mask,
            alpha=self.train_cfg.loss.tversky_alpha,
            beta=self.train_cfg.loss.tversky_beta,
        )
        return (1.0 - ti).mean()

    def _focal_tversky_loss(self, logits, mask):
        ti = _soft_tversky_index(
            logits,
            mask,
            alpha=self.train_cfg.loss.tversky_alpha,
            beta=self.train_cfg.loss.tversky_beta,
        )
        return torch.pow(1.0 - ti, self.train_cfg.loss.focal_gamma).mean()

    def _base_loss(self, logits, mask):
        loss_name = self.train_cfg.loss.name.lower()
        if loss_name == "bce":
            return self.bce(logits, mask)
        if loss_name == "dice":
            return self.dice(logits, mask)
        if loss_name == "bce_dice":
            return self.bce(logits, mask) + self.dice(logits, mask)
        if loss_name == "tversky":
            return self._tversky_loss(logits, mask)
        if loss_name == "bce_tversky":
            return self.bce(logits, mask) + self._tversky_loss(logits, mask)
        if loss_name == "focal_tversky":
            return self._focal_tversky_loss(logits, mask)
        if loss_name == "bce_focal_tversky":
            return self.bce(logits, mask) + self._focal_tversky_loss(logits, mask)
        raise ValueError(f"Unknown loss: {self.train_cfg.loss.name}")

    def _loss(self, logits, mask):
        loss = self._base_loss(logits, mask)
        # Optional topology term: composes with any base loss above rather than
        # needing its own named combination.
        weight = self.train_cfg.loss.cldice_weight
        if weight > 0:
            cldice = _soft_cldice(logits, mask, self.train_cfg.loss.cldice_iters)
            loss = loss + weight * (1.0 - cldice.mean())
        return loss

    def training_step(self, batch, batch_idx):
        # Weighted-sampling mode returns (img, mask, tile_index); static mode
        # returns (img, mask). Handle both so the sampler stays optional.
        if len(batch) == 3:
            img, mask, tile_index = batch
        else:
            img, mask = batch
            tile_index = None

        logits = self(img)
        loss = self._loss(logits, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if tile_index is None:
            return loss

        # Per-tile difficulty for hard-negative mining: per-sample BCE (a background
        # tile the model wrongly fires on scores high). Detached - purely a sampling
        # signal, not part of the optimized loss. HardNegativeMiningCallback collects
        # these and updates the sampler each epoch.
        with torch.no_grad():
            per_tile = F.binary_cross_entropy_with_logits(
                logits, mask, reduction="none"
            ).mean(dim=(1, 2, 3))
        return {
            "loss": loss,
            "tile_index": tile_index.detach().cpu(),
            "tile_difficulty": per_tile.detach().cpu(),
        }

    def on_validation_epoch_start(self):
        self._val_tp = 0.0
        self._val_fp = 0.0
        self._val_fn = 0.0

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self._loss(logits, mask)
        tp, fp, fn = _confusion_counts(logits, mask, self.train_cfg.threshold)
        self._val_tp += tp
        self._val_fp += fp
        self._val_fn += fn
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        stats = _stats_from_counts(
            self._val_tp,
            self._val_fp,
            self._val_fn,
            self.train_cfg.loss.tversky_alpha,
            self.train_cfg.loss.tversky_beta,
        )
        for k, v in stats.items():
            self.log(f"val/{k}", v, prog_bar=(k in {"dice", "iou", "tversky"}))
            self.log(f"val_{k}", v)

    def on_test_epoch_start(self):
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0

    def test_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self._loss(logits, mask)
        tp, fp, fn = _confusion_counts(logits, mask, self.train_cfg.threshold)
        self._test_tp += tp
        self._test_fp += fp
        self._test_fn += fn
        self.log("test/loss", loss, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        stats = _stats_from_counts(
            self._test_tp,
            self._test_fp,
            self._test_fn,
            self.train_cfg.loss.tversky_alpha,
            self.train_cfg.loss.tversky_beta,
        )
        for k, v in stats.items():
            self.log(f"test/{k}", v)
            self.log(f"test_{k}", v)

    def _build_optimizer(self):
        ratio = self.train_cfg.encoder_lr_ratio
        if ratio is None or not hasattr(self.model, "encoder"):
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.train_cfg.learning_rate,
                weight_decay=self.train_cfg.weight_decay,
            )

        encoder_params = list(self.model.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.train_cfg.learning_rate * ratio},
                {"params": other_params, "lr": self.train_cfg.learning_rate},
            ],
            weight_decay=self.train_cfg.weight_decay,
        )

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        scheduler_name = (self.train_cfg.scheduler.name or "none").lower()
        if scheduler_name in {"none", "off", "false"}:
            return optimizer
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.train_cfg.monitor_mode,
                factor=self.train_cfg.scheduler.factor,
                patience=self.train_cfg.scheduler.patience,
                min_lr=self.train_cfg.scheduler.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.train_cfg.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.train_cfg.max_epochs),
                eta_min=self.train_cfg.scheduler.min_lr,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        raise ValueError(f"Unknown scheduler: {self.train_cfg.scheduler.name}")
