"""Task-appropriate transforms.

Detection transforms operate jointly on ``(image, target)`` and keep boxes
consistent under flips/resizes. Classification uses standard ImageNet-style
pipelines (compatible with every timm model). Segmentation transforms apply
identical geometry to image and mask.
"""
from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T


# ------------------------------ detection --------------------------------- #
class DetCompose:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, img, target):
        for t in self.ts:
            img, target = t(img, target)
        return img, target


class DetToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class DetRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w = img.shape[-1] if torch.is_tensor(img) else img.size[0]
            img = F.hflip(img)
            if len(target["boxes"]):
                b = target["boxes"].clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]
                target["boxes"] = b
        return img, target


class DetColorJitter:
    """Photometric-only augmentation — safe for boxes; useful for PCB
    lighting/exposure variation across boards."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02):
        self.j = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target):
        return (self.j(img) if not torch.is_tensor(img) else img), target


def detection_transforms(train: bool) -> DetCompose:
    ts = []
    if train:
        ts += [DetColorJitter(), DetRandomHorizontalFlip()]
    ts.append(DetToTensor())
    return DetCompose(ts)


# ---------------------------- classification ------------------------------ #
def classification_transforms(train: bool, img_size: int = 224,
                              mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225)):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(0.2),          # components appear rotated
            T.ColorJitter(0.25, 0.25, 0.15, 0.03),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.15),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ----------------------------- segmentation ------------------------------- #
class SegJointTransform:
    """Applies identical resize/flip to image and mask; normalises image."""

    def __init__(self, train: bool, size: int = 512,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.train, self.size, self.mean, self.std = train, size, mean, std

    def __call__(self, img, mask):
        img = F.resize(img, [self.size, self.size])
        mask = F.resize(mask, [self.size, self.size],
                        interpolation=T.InterpolationMode.NEAREST)
        if self.train and random.random() < 0.5:
            img, mask = F.hflip(img), F.hflip(mask)
        if self.train and random.random() < 0.3:
            img = T.ColorJitter(0.25, 0.25, 0.15)(img)
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        mask = torch.as_tensor(__import__("numpy").array(mask), dtype=torch.long)
        return img, mask
