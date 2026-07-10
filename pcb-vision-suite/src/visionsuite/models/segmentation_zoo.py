"""Segmentation model zoo — 12 SOTA architectures behind one factory.

Semantic segmentation (pixel -> class):

=================  =============  ==========================================
registry key       framework      architecture
=================  =============  ==========================================
unet               smp            U-Net, ResNet34 encoder
unetplusplus       smp            U-Net++ (nested skip connections)
deeplabv3plus      smp            DeepLabV3+ ResNet50
fpn                smp            Feature Pyramid Network decoder
pspnet             smp            Pyramid Scene Parsing Network
manet              smp            Multi-scale Attention Net
linknet            smp            LinkNet (efficient)
pan                smp            Pyramid Attention Network
deeplabv3          torchvision    DeepLabV3 ResNet50
fcn                torchvision    Fully Convolutional Network ResNet50
lraspp             torchvision    LR-ASPP MobileNetV3 (edge/real-time)
segformer          huggingface    SegFormer-B0 (nvidia/mit-b0), transformer
=================  =============  ==========================================

Instance segmentation (per-object masks — the natural fit for PCB
components, using rectangular masks when only boxes exist):

=================  =============  ==========================================
maskrcnn           torchvision    Mask R-CNN ResNet50-FPN
maskrcnn_v2        torchvision    Mask R-CNN ResNet50-FPN v2
mask2former        huggingface    Mask2Former (facebook/mask2former-swin-tiny)
=================  =============  ==========================================

Every semantic model is normalised to ``forward(x) -> logits [B,C,H,W]`` via
thin adapters, so one engine trains them all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SegBundle:
    name: str
    framework: str
    kind: str                       # semantic | instance
    model: nn.Module


REGISTRY: dict[str, Callable] = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def list_models() -> list[str]:
    return sorted(REGISTRY)


def build_segmenter(name: str, num_classes: int, pretrained: bool = True,
                    **kw) -> SegBundle:
    if name not in REGISTRY:
        raise KeyError(f"Unknown segmenter '{name}'. Available: {list_models()}")
    return REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kw)


# ------------------------- segmentation_models_pytorch -------------------- #
SMP_ARCHS = {
    "unet": ("Unet", "resnet34"),
    "unetplusplus": ("UnetPlusPlus", "resnet34"),
    "deeplabv3plus": ("DeepLabV3Plus", "resnet50"),
    "fpn": ("FPN", "resnet34"),
    "pspnet": ("PSPNet", "resnet50"),
    "manet": ("MAnet", "resnet34"),
    "linknet": ("Linknet", "resnet34"),
    "pan": ("PAN", "resnet34"),
}


def _smp(name, num_classes, pretrained, encoder=None):
    import segmentation_models_pytorch as smp
    arch, default_enc = SMP_ARCHS[name]
    model = getattr(smp, arch)(
        encoder_name=encoder or default_enc,
        encoder_weights="imagenet" if pretrained else None,
        classes=num_classes,
    )
    return SegBundle(name, "smp", "semantic", model)


for _n in SMP_ARCHS:
    REGISTRY[_n] = (lambda _n: lambda num_classes, pretrained=True, encoder=None,
                    **kw: _smp(_n, num_classes, pretrained, encoder))(_n)


# ------------------------------ torchvision ------------------------------- #
class _TVSemanticAdapter(nn.Module):
    """torchvision seg models return {'out': logits}; normalise to logits."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]


def _tv_sem(name, ctor, head_swap, num_classes, pretrained):
    weights = "DEFAULT" if pretrained else None
    net = ctor(weights=weights, aux_loss=None) if name != "lraspp" \
        else ctor(weights=weights)
    head_swap(net, num_classes)
    return SegBundle(name, "torchvision", "semantic", _TVSemanticAdapter(net))


@register("deeplabv3")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead

    def swap(net, nc):
        net.classifier = DeepLabHead(2048, nc)
        net.aux_classifier = None
    return _tv_sem("deeplabv3", deeplabv3_resnet50, swap, num_classes, pretrained)


@register("fcn")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.segmentation import fcn_resnet50
    from torchvision.models.segmentation.fcn import FCNHead

    def swap(net, nc):
        net.classifier = FCNHead(2048, nc)
        net.aux_classifier = None
    return _tv_sem("fcn", fcn_resnet50, swap, num_classes, pretrained)


@register("lraspp")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    def swap(net, nc):
        low_c = net.classifier.low_classifier.in_channels
        high_c = net.classifier.high_classifier.in_channels
        net.classifier.low_classifier = nn.Conv2d(low_c, nc, 1)
        net.classifier.high_classifier = nn.Conv2d(high_c, nc, 1)
    return _tv_sem("lraspp", lraspp_mobilenet_v3_large, swap,
                   num_classes, pretrained)


# ------------------------------ HuggingFace ------------------------------- #
class _SegformerAdapter(nn.Module):
    def __init__(self, num_classes, checkpoint="nvidia/mit-b0"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        logits = self.net(pixel_values=x).logits          # 1/4 resolution
        return F.interpolate(logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


@register("segformer")
def _(num_classes, pretrained=True, checkpoint="nvidia/mit-b0", **kw):
    return SegBundle("segformer", "huggingface", "semantic",
                     _SegformerAdapter(num_classes, checkpoint))


# --------------------------- instance segmentation ------------------------ #
def _fix_maskrcnn(model, nc):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, nc)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, nc)


@register("maskrcnn")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    m = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    _fix_maskrcnn(m, num_classes + 1)
    return SegBundle("maskrcnn", "torchvision", "instance", m)


@register("maskrcnn_v2")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    m = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)
    _fix_maskrcnn(m, num_classes + 1)
    return SegBundle("maskrcnn_v2", "torchvision", "instance", m)


class _Mask2FormerAdapter(nn.Module):
    """Universal segmentation; exposed as instance model. Training uses HF
    loss; eval post-processes to per-instance masks."""

    def __init__(self, num_classes,
                 checkpoint="facebook/mask2former-swin-tiny-coco-instance"):
        super().__init__()
        from transformers import (AutoImageProcessor,
                                  Mask2FormerForUniversalSegmentation)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, images, targets=None):
        device = images[0].device
        if self.training:
            mask_labels = [t["masks"].float() for t in targets]
            class_labels = [t["labels"] - 1 for t in targets]
            enc = self.processor(images=[im.cpu() for im in images],
                                 return_tensors="pt", do_rescale=False)
            out = self.net(pixel_values=enc["pixel_values"].to(device),
                           mask_labels=[m.to(device) for m in mask_labels],
                           class_labels=[c.to(device) for c in class_labels])
            return {"loss": out.loss}
        enc = self.processor(images=[im.cpu() for im in images],
                             return_tensors="pt", do_rescale=False)
        out = self.net(pixel_values=enc["pixel_values"].to(device))
        sizes = [(im.shape[1], im.shape[2]) for im in images]
        segs = self.processor.post_process_instance_segmentation(
            out, target_sizes=sizes, threshold=0.5)
        results = []
        for s, (H, W) in zip(segs, sizes):
            seg_map = s["segmentation"]
            infos = s["segments_info"]
            masks, labels, scores = [], [], []
            for info in infos:
                masks.append((seg_map == info["id"]).to(torch.uint8))
                labels.append(info["label_id"] + 1)
                scores.append(info["score"])
            if masks:
                m = torch.stack(masks)
                ys = m.any(2); xs = m.any(1)
                boxes = []
                for i in range(len(m)):
                    yy = torch.where(ys[i])[0]; xx = torch.where(xs[i])[0]
                    if len(yy) and len(xx):
                        boxes.append([xx.min(), yy.min(), xx.max(), yy.max()])
                    else:
                        boxes.append([0, 0, 1, 1])
                results.append({
                    "masks": m.unsqueeze(1),
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                    "scores": torch.tensor(scores)})
            else:
                results.append({"masks": torch.zeros(0, 1, H, W),
                                "boxes": torch.zeros(0, 4),
                                "labels": torch.zeros(0, dtype=torch.int64),
                                "scores": torch.zeros(0)})
        return results


@register("mask2former")
def _(num_classes, pretrained=True, **kw):
    return SegBundle("mask2former", "huggingface", "instance",
                     _Mask2FormerAdapter(num_classes))
