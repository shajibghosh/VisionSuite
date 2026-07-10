"""Converters from every supported external annotation format into the
suite's canonical formats (COCO / classification manifest / mask pairs).

Supported inputs
----------------
- **COCO JSON** (passthrough / re-index)
- **Pascal VOC XML** (one XML per image)
- **YOLO txt** (``class cx cy w h`` normalised, + classes.txt/obj.names)
- **CSV** (columns: image, xmin, ymin, xmax, ymax, label — header
  auto-mapped; also plain ``image,label`` for classification)
- **LabelMe JSON** (polygons -> boxes + segmentation)
- **ImageFolder** (``root/class_x/img.jpg`` for classification)
- **Mask pairs** (``images/`` + ``masks/`` PNG for semantic segmentation)
- **PCB WACV-2019** (per-board folders with VOC-style XML, 31 classes)

Every converter returns a :class:`~.base.DatasetIR`; ``convert()`` then
serialises it to the canonical output for the requested task.
"""
from __future__ import annotations

import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from .base import (IMG_EXTS, DatasetIR, ImageRecord, Instance, detect_format,
                   ir_to_classification_manifest, ir_to_coco)


# --------------------------------------------------------------------------- #
# Individual format readers -> DatasetIR
# --------------------------------------------------------------------------- #
def read_coco(json_path: Path, image_root: Path | None = None) -> DatasetIR:
    data = json.loads(Path(json_path).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_img: dict[int, list] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    ir = DatasetIR(categories=[cats[k] for k in sorted(cats)])
    root = Path(image_root) if image_root else Path(json_path).parent
    for im in data["images"]:
        rec = ImageRecord(
            path=str((root / im["file_name"])),
            width=im["width"], height=im["height"],
        )
        for a in by_img.get(im["id"], []):
            seg = a.get("segmentation")
            poly = seg if isinstance(seg, list) and seg else None
            rec.instances.append(
                Instance(cats[a["category_id"]], tuple(a["bbox"]), poly,
                         a.get("iscrowd", 0))
            )
        ir.images.append(rec)
    return ir


def read_voc(root: Path) -> DatasetIR:
    """Pascal VOC: XML files either beside images or under Annotations/."""
    root = Path(root)
    xmls = sorted(root.rglob("*.xml"))
    ir = DatasetIR()
    for x in xmls:
        try:
            tree = ET.parse(x)
        except ET.ParseError:
            continue
        r = tree.getroot()
        if r.tag != "annotation":
            continue
        fname = (r.findtext("filename") or x.with_suffix(".jpg").name)
        img_path = _find_image(x.parent, fname) or _find_image(root, fname)
        size = r.find("size")
        if size is not None:
            w = int(float(size.findtext("width") or 0))
            h = int(float(size.findtext("height") or 0))
        else:
            w = h = 0
        if (not w or not h) and img_path:
            with Image.open(img_path) as im:
                w, h = im.size
        rec = ImageRecord(path=str(img_path or fname), width=w, height=h)
        for obj in r.findall("object"):
            name = obj.findtext("name") or "object"
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1 = float(bb.findtext("xmin") or 0); y1 = float(bb.findtext("ymin") or 0)
            x2 = float(bb.findtext("xmax") or 0); y2 = float(bb.findtext("ymax") or 0)
            rec.instances.append(Instance(name, (x1, y1, x2 - x1, y2 - y1)))
        ir.images.append(rec)
    return ir


def read_yolo(root: Path) -> DatasetIR:
    root = Path(root)
    names_file = next(
        (f for f in [root / "classes.txt", root / "obj.names",
                     root / "data" / "classes.txt"] if f.exists()), None)
    names = (names_file.read_text().split() if names_file else [])
    imgs = [f for f in root.rglob("*") if f.suffix.lower() in IMG_EXTS]
    ir = DatasetIR(categories=list(names))
    for img in sorted(imgs):
        txt = _yolo_label_for(img)
        with Image.open(img) as im:
            W, H = im.size
        rec = ImageRecord(path=str(img), width=W, height=H)
        if txt and txt.exists():
            for line in txt.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
                name = names[cid] if cid < len(names) else f"class_{cid}"
                rec.instances.append(Instance(
                    name, ((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H)))
        ir.images.append(rec)
    return ir


def _yolo_label_for(img: Path) -> Path | None:
    cand = img.with_suffix(".txt")
    if cand.exists():
        return cand
    if "images" in img.parts:
        parts = list(img.parts)
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return None


CSV_COL_ALIASES = {
    "image": {"image", "image_path", "filename", "file", "img", "image_name", "path"},
    "xmin": {"xmin", "x_min", "x1", "left"},
    "ymin": {"ymin", "y_min", "y1", "top"},
    "xmax": {"xmax", "x_max", "x2", "right"},
    "ymax": {"ymax", "y_max", "y2", "bottom"},
    "label": {"label", "class", "class_name", "category", "name", "type"},
}


def read_csv_annotations(csv_path: Path, image_root: Path | None = None) -> DatasetIR:
    """Detection CSV (image + box + label) or classification CSV (image + label)."""
    csv_path = Path(csv_path)
    root = Path(image_root) if image_root else csv_path.parent
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return DatasetIR()
    cols = {k.lower().strip(): k for k in rows[0].keys()}

    def col(name):
        for alias in CSV_COL_ALIASES[name]:
            if alias in cols:
                return cols[alias]
        return None

    c_img, c_lab = col("image"), col("label")
    boxes = all(col(k) for k in ("xmin", "ymin", "xmax", "ymax"))
    recs: dict[str, ImageRecord] = {}
    for r in rows:
        img_rel = r[c_img]
        if img_rel not in recs:
            p = root / img_rel
            w = h = 0
            if p.exists():
                with Image.open(p) as im:
                    w, h = im.size
            recs[img_rel] = ImageRecord(path=str(p), width=w, height=h)
        rec = recs[img_rel]
        if boxes:
            x1, y1 = float(r[col("xmin")]), float(r[col("ymin")])
            x2, y2 = float(r[col("xmax")]), float(r[col("ymax")])
            rec.instances.append(
                Instance(r[c_lab] if c_lab else "object",
                         (x1, y1, x2 - x1, y2 - y1)))
        elif c_lab:
            rec.label = r[c_lab]
    return DatasetIR(images=list(recs.values()))


def read_labelme(root: Path) -> DatasetIR:
    root = Path(root)
    ir = DatasetIR()
    for j in sorted(root.rglob("*.json")):
        try:
            data = json.loads(j.read_text())
        except Exception:
            continue
        if "shapes" not in data:
            continue
        img_path = j.parent / data.get("imagePath", j.with_suffix(".jpg").name)
        w = data.get("imageWidth", 0); h = data.get("imageHeight", 0)
        if (not w or not h) and img_path.exists():
            with Image.open(img_path) as im:
                w, h = im.size
        rec = ImageRecord(path=str(img_path), width=w, height=h)
        for s in data["shapes"]:
            pts = s.get("points", [])
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            poly = None
            if s.get("shape_type", "polygon") == "polygon" and len(pts) >= 3:
                flat = [v for p in pts for v in p]
                poly = [flat]
            rec.instances.append(
                Instance(s.get("label", "object"), (x1, y1, x2 - x1, y2 - y1), poly))
        ir.images.append(rec)
    return ir


def read_imagefolder(root: Path) -> DatasetIR:
    root = Path(root)
    ir = DatasetIR()
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                with Image.open(img) as im:
                    w, h = im.size
                ir.images.append(ImageRecord(str(img), w, h, label=cls_dir.name))
    return ir


def read_mask_pairs(root: Path) -> DatasetIR:
    """images/ + masks/ with matching stems; mask pixel value = class id."""
    root = Path(root)
    img_dir = next((root / n for n in ("images", "imgs") if (root / n).is_dir()), None)
    msk_dir = root / "masks"
    ir = DatasetIR()
    if not img_dir or not msk_dir.is_dir():
        return ir
    masks = {m.stem: m for m in msk_dir.iterdir() if m.suffix.lower() in IMG_EXTS}
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS or img.stem not in masks:
            continue
        with Image.open(img) as im:
            w, h = im.size
        ir.images.append(ImageRecord(str(img), w, h, mask_path=str(masks[img.stem])))
    return ir


def read_pcb_wacv(root: Path) -> DatasetIR:
    """WACV-2019 PCB dataset: one folder per board, each with the high-res
    board image and a VOC-style XML listing ~500 components (31 classes)."""
    ir = read_voc(Path(root))
    # Normalise class-name casing/whitespace, which varies across boards.
    for im in ir.images:
        for inst in im.instances:
            inst.category = inst.category.strip().lower().replace(" ", "_")
    return ir


def _find_image(folder: Path, name: str) -> Path | None:
    cand = folder / name
    if cand.exists():
        return cand
    stem = Path(name).stem
    for ext in IMG_EXTS:
        c = folder / f"{stem}{ext}"
        if c.exists():
            return c
    return None


READERS = {
    "coco": read_coco,
    "voc": read_voc,
    "yolo": read_yolo,
    "csv": read_csv_annotations,
    "labelme": read_labelme,
    "imagefolder": read_imagefolder,
    "mask_pairs": read_mask_pairs,
    "pcb_wacv": read_pcb_wacv,
}


# --------------------------------------------------------------------------- #
# Unified entrypoint
# --------------------------------------------------------------------------- #
def convert(source: str | Path, out_dir: str | Path, task: str,
            fmt: str | None = None) -> Path:
    """Convert *any* supported annotation source into the canonical format
    for ``task`` and return the canonical annotation path.

    detection      -> ``<out>/annotations.json`` (COCO)
    segmentation   -> instance: COCO; semantic: copied mask pairs
    classification -> ``<out>/manifest.csv`` + ``classes.txt``
    """
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = fmt or detect_format(source)
    if fmt == "unknown":
        raise ValueError(
            f"Could not auto-detect annotation format at {source}. "
            f"Pass fmt= one of {sorted(READERS)}")
    ir = READERS[fmt](source)

    if task == "classification":
        if fmt in ("coco", "voc", "yolo", "labelme", "pcb_wacv"):
            # Detection-style source: crop each instance into class folders.
            return crops_from_detections(ir, out_dir)
        return ir_to_classification_manifest(ir, out_dir)

    if task == "segmentation" and fmt == "mask_pairs":
        # Already canonical; write a manifest of pairs.
        manifest = out_dir / "seg_manifest.csv"
        with manifest.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["image_path", "mask_path"])
            for im in ir.images:
                w.writerow([im.path, im.mask_path])
        return manifest

    out_json = out_dir / "annotations.json"
    ir_to_coco(ir, out_json)
    return out_json


def crops_from_detections(ir: DatasetIR, out_dir: Path,
                          min_size: int = 8) -> Path:
    """Turn a detection IR into a classification set by cropping every box
    into ``out_dir/crops/<class>/``. This is how the PCB detection dataset
    becomes a 31-class component-classification dataset."""
    crop_root = out_dir / "crops"
    ir.ensure_categories()
    rows = []
    for im in ir.images:
        try:
            pil = Image.open(im.path).convert("RGB")
        except Exception:
            continue
        for k, inst in enumerate(im.instances):
            x, y, w, h = inst.bbox
            if w < min_size or h < min_size:
                continue
            crop = pil.crop((int(x), int(y), int(x + w), int(y + h)))
            cdir = crop_root / inst.category
            cdir.mkdir(parents=True, exist_ok=True)
            cpath = cdir / f"{Path(im.path).stem}_{k}.jpg"
            crop.save(cpath, quality=95)
            rows.append((str(cpath), inst.category))
    (out_dir / "classes.txt").write_text("\n".join(ir.categories))
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_path", "label"])
        w.writerows(rows)
    return manifest


def coco_to_yolo(coco_json: str | Path, out_dir: str | Path) -> Path:
    """Export canonical COCO to a YOLO/Ultralytics dataset (images symlinked,
    labels generated, ``data.yaml`` written). Needed by YOLO/RT-DETR trainers."""
    coco_json, out_dir = Path(coco_json), Path(out_dir)
    data = json.loads(coco_json.read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    img_dir = out_dir / "images"; lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    by_img: dict[int, list] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in data["images"]:
        src = Path(im["file_name"])
        dst = img_dir / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy(src, dst)
        lines = []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            W, H = im["width"], im["height"]
            lines.append(f"{id2idx[a['category_id']]} "
                         f"{(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}")
        (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines))
    yaml_path = out_dir / "data.yaml"
    names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(cats))
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\ntrain: images\nval: images\nnames:\n{names}\n")
    return yaml_path
