"""Canonical annotation schemas and automatic format detection.

Every converter in this package translates *some* external annotation format
into one of the suite's three canonical formats. All dataloaders read only
the canonical formats, which decouples the 30+ models from the dozens of
annotation formats found in the wild.

Canonical formats
-----------------
detection      -> COCO instances JSON  (``annotations.json`` + image dir)
classification -> manifest CSV ``image_path,label`` + ``classes.txt``
segmentation   -> semantic: paired PNG masks (pixel value = class id)
                  instance: COCO JSON with ``segmentation`` polygons/RLE

Format sniffing
---------------
``detect_format(path)`` inspects a directory/file and returns one of:
``coco``, ``voc``, ``yolo``, ``csv``, ``labelme``, ``imagefolder``,
``mask_pairs``, ``pcb_wacv``, or ``unknown``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Intermediate representation shared by all converters
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    """One annotated object: box in absolute xywh, optional polygon mask."""
    category: str
    bbox: tuple[float, float, float, float]          # x, y, w, h (pixels)
    polygon: Optional[list[list[float]]] = None       # [[x1,y1,x2,y2,...], ...]
    iscrowd: int = 0


@dataclass
class ImageRecord:
    path: str
    width: int
    height: int
    instances: list[Instance] = field(default_factory=list)
    label: Optional[str] = None                       # classification label
    mask_path: Optional[str] = None                   # semantic-seg mask


@dataclass
class DatasetIR:
    """Format-agnostic in-memory dataset all converters produce/consume."""
    images: list[ImageRecord] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def ensure_categories(self) -> None:
        if not self.categories:
            names = set()
            for im in self.images:
                names.update(i.category for i in im.instances)
                if im.label:
                    names.add(im.label)
            self.categories = sorted(names)


# --------------------------------------------------------------------------- #
# IR -> canonical COCO
# --------------------------------------------------------------------------- #
def ir_to_coco(ir: DatasetIR, out_json: str | Path) -> dict:
    """Serialise a :class:`DatasetIR` to a canonical COCO instances JSON."""
    ir.ensure_categories()
    cat_ids = {name: i + 1 for i, name in enumerate(ir.categories)}
    coco = {
        "info": {"description": "VisionSuite canonical COCO export"},
        "licenses": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for name, cid in cat_ids.items()
        ],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for img_id, im in enumerate(ir.images, start=1):
        coco["images"].append(
            {"id": img_id, "file_name": im.path, "width": im.width, "height": im.height}
        )
        for inst in im.instances:
            x, y, w, h = inst.bbox
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_ids[inst.category],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": inst.iscrowd,
            }
            if inst.polygon:
                ann["segmentation"] = inst.polygon
            coco["annotations"].append(ann)
            ann_id += 1
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    return coco


def ir_to_classification_manifest(ir: DatasetIR, out_dir: str | Path) -> Path:
    """Write ``manifest.csv`` (+ ``classes.txt``) for classification."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ir.ensure_categories()
    (out_dir / "classes.txt").write_text("\n".join(ir.categories))
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for im in ir.images:
            if im.label is not None:
                writer.writerow([im.path, im.label])
    return manifest


# --------------------------------------------------------------------------- #
# Format sniffing
# --------------------------------------------------------------------------- #
def _looks_like_coco(p: Path) -> bool:
    try:
        head = json.loads(p.read_text())
        return isinstance(head, dict) and {"images", "annotations"} <= set(head)
    except Exception:
        return False


def _looks_like_labelme_dir(d: Path) -> bool:
    for j in list(d.glob("*.json"))[:5]:
        try:
            data = json.loads(j.read_text())
            if "shapes" in data and "imagePath" in data:
                return True
        except Exception:
            continue
    return False


def detect_format(path: str | Path) -> str:
    """Best-effort detection of the annotation format at ``path``."""
    p = Path(path)
    if p.is_file():
        if p.suffix == ".json":
            if _looks_like_coco(p):
                return "coco"
            try:
                if "shapes" in json.loads(p.read_text()):
                    return "labelme"
            except Exception:
                pass
            return "unknown"
        if p.suffix == ".csv":
            return "csv"
        return "unknown"

    if not p.is_dir():
        return "unknown"

    # Directory heuristics, roughly from most to least specific.
    if any(p.rglob("*.xml")):
        # VOC-style XML; the PCB WACV zip also ships VOC XMLs but with a
        # recognisable layout (per-board folders). Prefer specific detector.
        if (p / "Annotations").is_dir() or (p / "JPEGImages").is_dir():
            return "voc"
        xml = next(p.rglob("*.xml"))
        if "<annotation" in xml.read_text(errors="ignore")[:2000]:
            return "pcb_wacv" if _looks_like_pcb_wacv(p) else "voc"
    for j in p.glob("*.json"):
        if _looks_like_coco(j):
            return "coco"
    if _looks_like_labelme_dir(p):
        return "labelme"
    if (p / "classes.txt").exists() or (p / "obj.names").exists() or any(
        _txt_is_yolo(t) for t in list(p.rglob("*.txt"))[:10]
    ):
        return "yolo"
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if subdirs and all(
        any(f.suffix.lower() in IMG_EXTS for f in d.iterdir() if f.is_file())
        for d in subdirs[:5]
    ):
        # images/ + masks/ pair => semantic segmentation
        names = {d.name.lower() for d in subdirs}
        if {"images", "masks"} <= names or {"imgs", "masks"} <= names:
            return "mask_pairs"
        return "imagefolder"
    if any(p.glob("*.csv")):
        return "csv"
    return "unknown"


def _txt_is_yolo(t: Path) -> bool:
    try:
        line = t.read_text().strip().splitlines()[0].split()
        return len(line) >= 5 and all(_is_float(x) for x in line) \
            and float(line[0]).is_integer()
    except Exception:
        return False


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _looks_like_pcb_wacv(p: Path) -> bool:
    """The WACV-2019 PCB zip nests one folder per board, each containing the
    board image plus a VOC-style XML with 31 component classes."""
    boards = [d for d in p.iterdir() if d.is_dir()]
    hits = 0
    for b in boards[:10]:
        if any(b.glob("*.xml")) and any(
            f.suffix.lower() in IMG_EXTS for f in b.iterdir() if f.is_file()
        ):
            hits += 1
    return hits >= 2
