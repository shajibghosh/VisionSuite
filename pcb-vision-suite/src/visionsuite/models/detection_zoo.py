"""Detection model zoo — 14 SOTA architectures behind one factory.

===================  =============  =========================================
registry key         framework      architecture
===================  =============  =========================================
faster_rcnn          torchvision    Faster R-CNN ResNet50-FPN
faster_rcnn_v2       torchvision    Faster R-CNN ResNet50-FPN v2 (2023 recipe)
retinanet            torchvision    RetinaNet ResNet50-FPN
retinanet_v2         torchvision    RetinaNet v2 (improved training recipe)
fcos                 torchvision    FCOS anchor-free ResNet50-FPN
ssd300               torchvision    SSD300 VGG16
ssdlite              torchvision    SSDLite MobileNetV3-Large (edge)
detr                 huggingface    DETR ResNet-50 (facebook/detr-resnet-50)
deformable_detr      huggingface    Deformable DETR (SenseTime)
conditional_detr     huggingface    Conditional DETR ResNet-50
yolos                huggingface    YOLOS-small (ViT detector)
yolov8               ultralytics    YOLOv8 (n/s/m/l/x via ``variant``)
yolo11               ultralytics    YOLO11 (latest Ultralytics family)
rtdetr               ultralytics    RT-DETR-L (real-time DETR)
===================  =============  =========================================

Frameworks differ in training APIs, so builders return a ``ModelBundle``
that records which engine drives it:

- ``torchvision`` models train through :mod:`visionsuite.engine.detection_engine`
  (they natively take ``(images, targets)`` and return a loss dict).
- ``huggingface`` DETR-family models are wrapped in :class:`HFDetrAdapter`
  so they expose the *same* torchvision-style API and reuse the same engine.
- ``ultralytics`` models train through Ultralytics' own trainer (invoked by
  ``scripts/train_detection.py`` after auto-exporting COCO -> YOLO format);
  metrics/plots are harvested back into the suite's unified run directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    name: str
    framework: str                 # torchvision | huggingface | ultralytics
    model: object | None = None    # nn.Module (None until built for ultralytics)
    build_info: dict | None = None


REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def list_models() -> list[str]:
    return sorted(REGISTRY)


def build_detector(name: str, num_classes: int, pretrained: bool = True,
                   variant: str | None = None, **kw) -> ModelBundle:
    """``num_classes`` EXCLUDES background; builders adjust per framework."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown detector '{name}'. Available: {list_models()}")
    return REGISTRY[name](num_classes=num_classes, pretrained=pretrained,
                          variant=variant, **kw)


# --------------------------------------------------------------------------- #
# torchvision family
# --------------------------------------------------------------------------- #
def _tv(name, ctor, head_fix, num_classes, pretrained):
    import torchvision
    weights = "DEFAULT" if pretrained else None
    model = ctor(weights=weights)
    head_fix(model, num_classes + 1)  # +1 background
    return ModelBundle(name, "torchvision", model)


def _fix_frcnn(model, nc):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, nc)


def _fix_retina(model, nc):
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    head = model.head.classification_head
    in_ch = head.conv[0][0].in_channels if isinstance(head.conv[0], nn.Sequential) \
        else head.conv[0].in_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head.classification_head = RetinaNetClassificationHead(
        in_ch, num_anchors, nc, norm_layer=lambda c: nn.GroupNorm(32, c))


def _fix_fcos(model, nc):
    from torchvision.models.detection.fcos import FCOSClassificationHead
    in_ch = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head.classification_head = FCOSClassificationHead(in_ch, num_anchors, nc)


def _fix_ssd(model, nc):
    import torchvision
    from torchvision.models.detection.ssd import SSDClassificationHead
    in_ch = torchvision.models.detection._utils.retrieve_out_channels(
        model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_ch, num_anchors, nc)


def _fix_ssdlite(model, nc):
    import torchvision
    from functools import partial
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
    in_ch = torchvision.models.detection._utils.retrieve_out_channels(
        model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_ch, num_anchors, nc, norm_layer=partial(nn.BatchNorm2d, eps=0.001))


@register("faster_rcnn")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    return _tv("faster_rcnn", fasterrcnn_resnet50_fpn, _fix_frcnn,
               num_classes, pretrained)


@register("faster_rcnn_v2")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    return _tv("faster_rcnn_v2", fasterrcnn_resnet50_fpn_v2, _fix_frcnn,
               num_classes, pretrained)


@register("retinanet")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import retinanet_resnet50_fpn
    return _tv("retinanet", retinanet_resnet50_fpn, _fix_retina,
               num_classes, pretrained)


@register("retinanet_v2")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    return _tv("retinanet_v2", retinanet_resnet50_fpn_v2, _fix_retina,
               num_classes, pretrained)


@register("fcos")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import fcos_resnet50_fpn
    return _tv("fcos", fcos_resnet50_fpn, _fix_fcos, num_classes, pretrained)


@register("ssd300")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import ssd300_vgg16
    return _tv("ssd300", ssd300_vgg16, _fix_ssd, num_classes, pretrained)


@register("ssdlite")
def _(num_classes, pretrained=True, **kw):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    return _tv("ssdlite", ssdlite320_mobilenet_v3_large, _fix_ssdlite,
               num_classes, pretrained)


# --------------------------------------------------------------------------- #
# HuggingFace DETR family — wrapped to look like torchvision detectors
# --------------------------------------------------------------------------- #
HF_CHECKPOINTS = {
    "detr": "facebook/detr-resnet-50",
    "deformable_detr": "SenseTime/deformable-detr",
    "conditional_detr": "microsoft/conditional-detr-resnet-50",
    "yolos": "hustvl/yolos-small",
}


class HFDetrAdapter(nn.Module):
    """Adapts a HF object-detection model to the torchvision detector API:

    train:  ``adapter(images, targets) -> {loss_name: tensor}``
    eval:   ``adapter(images) -> [{'boxes','scores','labels'}, ...]``

    Boxes are converted torchvision-xyxy <-> DETR normalised cxcywh, and
    labels shifted (torchvision reserves 0 for background; DETR does not).
    """

    def __init__(self, checkpoint: str, num_classes: int):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.net = AutoModelForObjectDetection.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, images, targets=None):
        device = images[0].device
        pixel = [im for im in images]
        if self.training:
            labels = []
            for im, t in zip(images, targets):
                _, H, W = im.shape
                b = t["boxes"]
                cxcywh = torch.stack([
                    (b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H,
                    (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H], dim=1)
                labels.append({"class_labels": t["labels"] - 1,
                               "boxes": cxcywh})
            enc = self.processor(images=[im.cpu() for im in pixel],
                                 return_tensors="pt", do_rescale=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.net(**enc, labels=labels)
            return {"loss": out.loss}
        enc = self.processor(images=[im.cpu() for im in pixel],
                             return_tensors="pt", do_rescale=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.net(**enc)
        sizes = torch.tensor([[im.shape[1], im.shape[2]] for im in images],
                             device=device)
        results = self.processor.post_process_object_detection(
            out, threshold=0.0, target_sizes=sizes)
        return [{"boxes": r["boxes"], "scores": r["scores"],
                 "labels": r["labels"] + 1} for r in results]


def _hf(name, num_classes, **kw):
    return ModelBundle(name, "huggingface",
                       HFDetrAdapter(HF_CHECKPOINTS[name], num_classes))


for _n in HF_CHECKPOINTS:
    REGISTRY[_n] = (lambda _n: lambda num_classes, pretrained=True, **kw:
                    _hf(_n, num_classes))(_n)


# --------------------------------------------------------------------------- #
# Ultralytics family (trained via Ultralytics' own engine)
# --------------------------------------------------------------------------- #
ULTRA_DEFAULT_WEIGHTS = {
    "yolov8": "yolov8{v}.pt",
    "yolo11": "yolo11{v}.pt",
    "rtdetr": "rtdetr-l.pt",
}


def _ultra(name, variant, **kw):
    v = variant or ("s" if name != "rtdetr" else "")
    weights = ULTRA_DEFAULT_WEIGHTS[name].format(v=v)
    return ModelBundle(name, "ultralytics", None,
                       build_info={"weights": weights})


for _n in ULTRA_DEFAULT_WEIGHTS:
    REGISTRY[_n] = (lambda _n: lambda num_classes, pretrained=True,
                    variant=None, **kw: _ultra(_n, variant))(_n)
