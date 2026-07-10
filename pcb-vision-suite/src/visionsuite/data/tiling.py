"""High-resolution image tiling (SAHI-style) for the PCB use case.

PCB board photos are 15+ megapixels with ~500 small components each. Feeding
them whole to a detector destroys small-object recall, so we:

1. **tile_coco_dataset** — pre-slice every image into overlapping tiles
   (default 1024px, 20% overlap), clipping/remapping the boxes, and emit a
   new canonical COCO file. Training then proceeds exactly as with any other
   dataset.
2. **stitch_predictions** — at inference time, run the model per-tile and
   merge tile detections back into full-board coordinates with global NMS.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _tile_grid(W: int, H: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


def tile_coco_dataset(ann_file: str | Path, out_dir: str | Path,
                      tile: int = 1024, overlap: float = 0.2,
                      min_visibility: float = 0.4,
                      keep_empty: bool = False) -> Path:
    """Slice a canonical COCO dataset into tiles; returns new COCO json path."""
    ann_file, out_dir = Path(ann_file), Path(out_dir)
    img_out = out_dir / "tiles"
    img_out.mkdir(parents=True, exist_ok=True)
    data = json.loads(ann_file.read_text())
    by_img: dict[int, list] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    new = {"info": {"description": f"tiled {tile}px / {overlap:.0%} overlap"},
           "categories": data["categories"], "images": [], "annotations": []}
    nid = aid = 1
    for im in data["images"]:
        src = Path(im["file_name"])
        pil = Image.open(src).convert("RGB")
        W, H = pil.size
        for tx, ty in _tile_grid(W, H, tile, overlap):
            tw, th = min(tile, W - tx), min(tile, H - ty)
            anns = []
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                ix1, iy1 = max(x, tx), max(y, ty)
                ix2, iy2 = min(x + w, tx + tw), min(y + h, ty + th)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 1 or ih <= 1:
                    continue
                if (iw * ih) / max(w * h, 1e-6) < min_visibility:
                    continue
                anns.append({"id": aid, "image_id": nid,
                             "category_id": a["category_id"],
                             "bbox": [ix1 - tx, iy1 - ty, iw, ih],
                             "area": iw * ih, "iscrowd": a.get("iscrowd", 0)})
                aid += 1
            if not anns and not keep_empty:
                continue
            tpath = img_out / f"{src.stem}_x{tx}_y{ty}.jpg"
            if not tpath.exists():
                pil.crop((tx, ty, tx + tw, ty + th)).save(tpath, quality=95)
            new["images"].append({"id": nid, "file_name": str(tpath),
                                  "width": tw, "height": th})
            new["annotations"].extend(anns)
            nid += 1
    out_json = out_dir / "annotations_tiled.json"
    out_json.write_text(json.dumps(new))
    return out_json


@torch.no_grad()
def stitch_predictions(predict_fn, image: Image.Image, tile: int = 1024,
                       overlap: float = 0.2, iou_thr: float = 0.55,
                       score_thr: float = 0.05) -> dict:
    """Tile ``image``, run ``predict_fn(tile_pil) -> {boxes,scores,labels}``
    per tile, shift boxes back to board coordinates, and apply global NMS."""
    from torchvision.ops import batched_nms
    W, H = image.size
    boxes, scores, labels = [], [], []
    for tx, ty in _tile_grid(W, H, tile, overlap):
        crop = image.crop((tx, ty, min(tx + tile, W), min(ty + tile, H)))
        out = predict_fn(crop)
        b = out["boxes"]
        if len(b) == 0:
            continue
        b = b.clone()
        b[:, [0, 2]] += tx
        b[:, [1, 3]] += ty
        boxes.append(b); scores.append(out["scores"]); labels.append(out["labels"])
    if not boxes:
        z = torch.zeros(0)
        return {"boxes": z.reshape(0, 4), "scores": z, "labels": z.long()}
    boxes = torch.cat(boxes); scores = torch.cat(scores); labels = torch.cat(labels)
    keep = scores >= score_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = batched_nms(boxes, scores, labels, iou_thr)
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
