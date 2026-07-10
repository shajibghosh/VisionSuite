"""PyTorch datasets that read only the canonical formats.

- :class:`CocoDetection`     — COCO json -> (image tensor, target dict)
- :class:`ManifestClassification` — manifest.csv -> (image tensor, label id)
- :class:`SemanticSegmentation`   — image/mask pairs -> (image, mask)
- :class:`CocoInstanceSegmentation` — COCO polygons -> Mask R-CNN targets

All detection targets follow the torchvision convention
(``boxes`` xyxy FloatTensor[N,4], ``labels`` Int64Tensor[N]) so a single
training engine drives every torchvision/HF model. High-resolution PCB
boards are handled by :mod:`visionsuite.data.tiling`, which pre-slices
images into overlapping tiles before training.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None  # PCB boards are 15+ MP


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class CocoDetection(Dataset):
    """Canonical COCO detection dataset.

    Returns ``(image, target)`` where target has ``boxes`` (xyxy), ``labels``
    (1-based to keep torchvision's background=0 convention), ``image_id``,
    ``area`` and ``iscrowd`` — directly consumable by every torchvision
    detector and easily adapted for DETR-family models.
    """

    def __init__(self, ann_file: str | Path, transforms=None,
                 image_ids: list[int] | None = None):
        self.data = json.loads(Path(ann_file).read_text())
        self.transforms = transforms
        self.cats = {c["id"]: c["name"]
                     for c in sorted(self.data["categories"], key=lambda c: c["id"])}
        # remap category ids -> contiguous 1..K
        self.cat_remap = {cid: i + 1 for i, cid in enumerate(sorted(self.cats))}
        self.class_names = [self.cats[c] for c in sorted(self.cats)]
        self.anns_by_img: dict[int, list] = {}
        for a in self.data["annotations"]:
            self.anns_by_img.setdefault(a["image_id"], []).append(a)
        imgs = self.data["images"]
        if image_ids is not None:
            keep = set(image_ids)
            imgs = [im for im in imgs if im["id"] in keep]
        self.images = imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = Image.open(info["file_name"]).convert("RGB")
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in self.anns_by_img.get(info["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_remap[a["category_id"]])
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    @staticmethod
    def collate(batch):
        return tuple(zip(*batch))


def split_coco(ann_file: str | Path, val_frac: float = 0.2, seed: int = 42
               ) -> tuple[list[int], list[int]]:
    """Deterministic image-level train/val split of a COCO file."""
    data = json.loads(Path(ann_file).read_text())
    ids = [im["id"] for im in data["images"]]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
class ManifestClassification(Dataset):
    """Reads ``manifest.csv`` (image_path,label) + ``classes.txt``."""

    def __init__(self, manifest: str | Path, transform=None,
                 indices: list[int] | None = None):
        manifest = Path(manifest)
        classes_file = manifest.parent / "classes.txt"
        with manifest.open() as f:
            self.rows = list(csv.DictReader(f))
        if classes_file.exists():
            self.class_names = classes_file.read_text().split("\n")
            self.class_names = [c for c in self.class_names if c.strip()]
        else:
            self.class_names = sorted({r["label"] for r in self.rows})
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[r["label"]]

    def labels(self) -> list[int]:
        return [self.class_to_idx[r["label"]] for r in self.rows]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights — essential for the heavily skewed PCB
        component distribution (mostly resistors/capacitors)."""
        counts = np.bincount(self.labels(), minlength=len(self.class_names))
        w = counts.sum() / np.maximum(counts, 1) / len(self.class_names)
        return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
class SemanticSegmentation(Dataset):
    """image/mask PNG pairs from ``seg_manifest.csv``; mask value = class id."""

    def __init__(self, manifest: str | Path, num_classes: int,
                 transform=None, indices: list[int] | None = None):
        with Path(manifest).open() as f:
            self.rows = list(csv.DictReader(f))
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        mask = Image.open(r["mask_path"])
        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
            mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


class CocoInstanceSegmentation(CocoDetection):
    """COCO with polygon masks -> targets including ``masks`` for Mask R-CNN.

    If the COCO file has no ``segmentation`` fields (pure detection, as in the
    PCB dataset), boxes are used as rectangular masks so instance-seg models
    can still be fine-tuned as strong detectors with mask heads.
    """

    def __getitem__(self, idx):
        info = self.images[idx]
        img = Image.open(info["file_name"]).convert("RGB")
        W, H = img.size
        boxes, labels, masks = [], [], []
        for a in self.anns_by_img.get(info["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_remap[a["category_id"]])
            m = np.zeros((H, W), dtype=np.uint8)
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                from PIL import ImageDraw
                mm = Image.new("L", (W, H), 0)
                d = ImageDraw.Draw(mm)
                for poly in seg:
                    pts = list(zip(poly[0::2], poly[1::2]))
                    if len(pts) >= 3:
                        d.polygon(pts, outline=1, fill=1)
                m = np.array(mm, dtype=np.uint8)
            else:
                m[int(y):int(y + h), int(x):int(x + w)] = 1
            masks.append(m)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks) if masks
                                     else np.zeros((0, H, W)), dtype=torch.uint8),
            "image_id": torch.tensor([info["id"]]),
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target
