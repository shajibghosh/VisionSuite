"""Classification model zoo — 12 SOTA architectures via a single factory.

All models are created through `timm <https://github.com/huggingface/pytorch-image-models>`_,
which gives us pretrained weights, a uniform ``forward``, and
``num_classes`` head replacement for free. The registry maps a short,
stable suite name to a concrete timm checkpoint; pass any other timm model
name directly and it also works (``build_classifier("timm/<name>")``).

=================  ==========================================================
registry key       timm checkpoint
=================  ==========================================================
resnet50           resnet50.a1_in1k
convnext           convnext_tiny.fb_in22k_ft_in1k
convnext_v2        convnextv2_tiny.fcmae_ft_in22k_in1k
vit                vit_base_patch16_224.augreg2_in21k_ft_in1k
deit3              deit3_small_patch16_224.fb_in22k_ft_in1k
swin               swin_tiny_patch4_window7_224.ms_in22k_ft_in1k
swin_v2            swinv2_tiny_window8_256.ms_in1k
efficientnet       tf_efficientnetv2_s.in21k_ft_in1k
maxvit             maxvit_tiny_tf_224.in1k
regnety            regnety_016.tv2_in1k
mobilenetv3        mobilenetv3_large_100.ra_in1k
densenet           densenet121.ra_in1k
eva02              eva02_small_patch14_336.mim_in22k_ft_in1k
=================  ==========================================================
"""
from __future__ import annotations

import torch.nn as nn

REGISTRY: dict[str, str] = {
    "resnet50": "resnet50.a1_in1k",
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_v2": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit3": "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "swin": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_v2": "swinv2_tiny_window8_256.ms_in1k",
    "efficientnet": "tf_efficientnetv2_s.in21k_ft_in1k",
    "maxvit": "maxvit_tiny_tf_224.in1k",
    "regnety": "regnety_016.tv2_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    "densenet": "densenet121.ra_in1k",
    "eva02": "eva02_small_patch14_336.mim_in22k_ft_in1k",
}


def list_models() -> list[str]:
    return sorted(REGISTRY)


def build_classifier(name: str, num_classes: int, pretrained: bool = True,
                     freeze_backbone: bool = False) -> nn.Module:
    """Create any registered (or raw ``timm/<name>``) classifier with a fresh
    ``num_classes``-way head. ``freeze_backbone=True`` fine-tunes the head
    only — a strong baseline for the small PCB crop dataset."""
    import timm
    timm_name = REGISTRY.get(name, name.removeprefix("timm/"))
    model = timm.create_model(timm_name, pretrained=pretrained,
                              num_classes=num_classes)
    if freeze_backbone:
        head = model.get_classifier()
        head_params = set(id(p) for p in head.parameters())
        for p in model.parameters():
            if id(p) not in head_params:
                p.requires_grad_(False)
    return model


def resolve_input_size(model) -> int:
    cfg = getattr(model, "pretrained_cfg", None) or {}
    size = cfg.get("input_size", (3, 224, 224))
    return size[-1]


def resolve_norm(model) -> tuple[tuple, tuple]:
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return (cfg.get("mean", (0.485, 0.456, 0.406)),
            cfg.get("std", (0.229, 0.224, 0.225)))


def param_groups_lrd(model, base_lr: float, weight_decay: float = 0.05,
                     layer_decay: float = 0.75):
    """Layer-wise learning-rate decay for transformer backbones (ViT/Swin/
    DeiT/EVA). Falls back to two groups (decay / no-decay) for CNNs."""
    no_decay = {"bias", "bn", "norm", "ln", "pos_embed", "cls_token"}
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay if any(k in n.lower() for k in no_decay) else decay).append(p)
        return [{"params": decay, "lr": base_lr, "weight_decay": weight_decay},
                {"params": nodecay, "lr": base_lr, "weight_decay": 0.0}]
    n_layers = len(blocks) + 1
    scales = [layer_decay ** (n_layers - i) for i in range(n_layers + 1)]

    def layer_id(name: str) -> int:
        if name.startswith(("cls_token", "pos_embed", "patch_embed")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return n_layers

    groups: dict[tuple, dict] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(n)
        wd = 0.0 if any(k in n.lower() for k in no_decay) else weight_decay
        key = (lid, wd)
        g = groups.setdefault(key, {"params": [], "lr": base_lr * scales[lid],
                                    "weight_decay": wd})
        g["params"].append(p)
    return list(groups.values())
