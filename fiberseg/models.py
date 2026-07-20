# models.py

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

from .config import ModelConfig


def create_model(cfg: ModelConfig):
    """Create an exchangeable segmentation model.

    Supported:
      - segmentation_models_pytorch models with encoder_weights=None or "imagenet"
      - NASA pretrained microscopy models with encoder_weights="micronet"

    Important:
      All returned models must output RAW LOGITS, not sigmoid probabilities.
    """

    encoder_weights = cfg.encoder_weights

    if isinstance(encoder_weights, str) and encoder_weights.lower() == "micronet":
        return create_micronet_model(cfg)

    return create_smp_model(cfg)


def create_smp_model(cfg: ModelConfig):
    if not hasattr(smp, cfg.architecture):
        available = [x for x in dir(smp) if x[:1].isupper()]
        raise ValueError(
            f"Unknown SMP architecture {cfg.architecture!r}. "
            f"Examples: {available[:20]}"
        )

    cls = getattr(smp, cfg.architecture)

    return cls(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        activation=None,
    )


def create_micronet_model(cfg: ModelConfig):
    if cfg.in_channels != 3:
        raise ValueError(
            "For the MicroNet experiment, use in_channels: 3 and "
            "replicate grayscale SEM images to 3 channels in the dataset."
        )

    try:
        import pretrained_microscopy_models as pmm
    except ImportError as e:
        raise ImportError(
            "encoder_weights='micronet' requires NASA pretrained-microscopy-models. "
            "Install it with:\n"
            "pip install git+https://github.com/nasa/pretrained-microscopy-models"
        ) from e

    model = pmm.segmentation_training.create_segmentation_model(
        cfg.architecture,
        cfg.encoder_name,
        "micronet",
        classes=cfg.classes,
    )

    force_raw_logits(model)

    return model


def force_raw_logits(model: nn.Module) -> None:
    """Remove common output activations so the model returns raw logits.

    This is needed because the NASA MicroNet helper may create a segmentation
    model with an output activation. The rest of this training pipeline expects
    raw logits and applies sigmoid only inside the loss/metrics/inference code.
    """

    # segmentation_models_pytorch SegmentationHead is usually:
    #   Conv2d -> Upsampling/Identity -> Activation
    if hasattr(model, "segmentation_head"):
        head = model.segmentation_head

        # Replace explicit .activation if present.
        if hasattr(head, "activation"):
            head.activation = nn.Identity()

        # Replace common final activation modules inside Sequential heads.
        if isinstance(head, nn.Sequential):
            for i, module in enumerate(head):
                if _is_output_activation(module):
                    head[i] = nn.Identity()

        # Some SMP heads store activation as the last module.
        children = list(head.children())
        if children:
            last_name, last_module = list(head.named_children())[-1]
            if _is_output_activation(last_module):
                setattr(head, last_name, nn.Identity())

    # Extra recursive safety pass.
    for name, module in model.named_modules():
        if name.endswith("activation") and _is_output_activation(module):
            parent, child_name = _get_parent_module(model, name)
            if parent is not None:
                setattr(parent, child_name, nn.Identity())


def _is_output_activation(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()

    if isinstance(module, (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)):
        return True

    # segmentation_models_pytorch uses an Activation wrapper.
    if name == "activation":
        return True

    return False


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")

    if len(parts) == 1:
        return model, parts[0]

    parent = model

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    return parent, parts[-1]