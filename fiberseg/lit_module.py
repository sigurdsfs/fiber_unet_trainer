# lit_module.py

from __future__ import annotations

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .config import ModelConfig, TrainConfig, to_dict
from .models import create_model


def _binary_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    alpha: float = 0.3,
    beta: float = 0.7,
):
    pred = torch.sigmoid(logits) > threshold
    targ = target > 0.5
    tp = (pred & targ).sum().float()
    fp = (pred & ~targ).sum().float()
    fn = (~pred & targ).sum().float()
    eps = torch.tensor(1e-8, device=logits.device)
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

    def _loss(self, logits, mask):
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

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self._loss(logits, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self._loss(logits, mask)
        stats = _binary_stats(
            logits,
            mask,
            self.train_cfg.threshold,
            self.train_cfg.loss.tversky_alpha,
            self.train_cfg.loss.tversky_beta,
        )
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True)
        for k, v in stats.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=(k in {"dice", "iou", "tversky"}))
            self.log(f"val_{k}", v, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self._loss(logits, mask)
        stats = _binary_stats(
            logits,
            mask,
            self.train_cfg.threshold,
            self.train_cfg.loss.tversky_alpha,
            self.train_cfg.loss.tversky_beta,
        )
        self.log("test/loss", loss, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)
        for k, v in stats.items():
            self.log(f"test/{k}", v, on_epoch=True)
            self.log(f"test_{k}", v, on_epoch=True)
        return loss

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
                mode="max",
                factor=self.train_cfg.scheduler.factor,
                patience=self.train_cfg.scheduler.patience,
                min_lr=self.train_cfg.scheduler.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/tversky",
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
