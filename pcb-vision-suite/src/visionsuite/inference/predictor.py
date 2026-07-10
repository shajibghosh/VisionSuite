"""Unified inference + reporting.

``Predictor.from_run(run_dir)`` reloads any suite checkpoint (task, model
name, class names and config are stored inside ``best.pt``) and exposes:

- ``predict_image(path)``       — one image -> structured result
- ``predict_batch(paths|dir)``  — many images -> results + summary report

Detection/instance results include, per object: class name, confidence,
bbox (xyxy), **pixel area**, relative area (% of image) and centroid.
Reports are written as ``results.json`` + ``results.csv`` + annotated
images + a per-class summary (counts, mean confidence, total area).

High-resolution images are automatically tiled
(:func:`visionsuite.data.tiling.stitch_predictions`) when larger than
``tile_threshold`` pixels on a side — essential for 15 MP PCB boards.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..data.tiling import stitch_predictions
from ..data.transforms import classification_transforms
from ..models.classification_zoo import (build_classifier, resolve_input_size,
                                         resolve_norm)
from ..models.detection_zoo import build_detector
from ..models.segmentation_zoo import build_segmenter

Image.MAX_IMAGE_PIXELS = None
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
           (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]


class Predictor:
    def __init__(self, task: str, model, class_names: list[str],
                 device: str | None = None, cfg: dict | None = None):
        self.task, self.class_names, self.cfg = task, class_names, cfg or {}
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        if task == "classification":
            self._size = self.cfg.get("img_size") or resolve_input_size(model)
            mean, std = resolve_norm(model)
            self._tf = classification_transforms(False, self._size, mean, std)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_run(cls, run_dir: str | Path, device: str | None = None,
                 which: str = "best") -> "Predictor":
        run_dir = Path(run_dir)
        task = run_dir.parent.name
        ckpt = torch.load(run_dir / "checkpoints" / f"{which}.pt",
                          map_location="cpu", weights_only=False)
        cfg, names = ckpt["cfg"], ckpt["class_names"]
        n = len(names)
        if task == "detection":
            model = build_detector(cfg["model"], n, pretrained=False).model
        elif task == "classification":
            model = build_classifier(cfg["model"], n, pretrained=False)
        else:
            model = build_segmenter(cfg["model"], n, pretrained=False).model
        model.load_state_dict(ckpt["model"])
        return cls(task, model, names, device, cfg)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_image(self, image: str | Path | Image.Image,
                      score_thr: float = 0.35, tile_threshold: int = 2048,
                      tile: int = 1024) -> dict:
        pil = image if isinstance(image, Image.Image) \
            else Image.open(image).convert("RGB")
        t0 = time.time()
        if self.task == "classification":
            out = self._predict_cls(pil)
        elif self.task == "detection" or self._is_instance():
            out = self._predict_det(pil, score_thr, tile_threshold, tile)
        else:
            out = self._predict_semantic(pil)
        out["latency_s"] = round(time.time() - t0, 4)
        out["image_size"] = list(pil.size)
        return out

    def _is_instance(self) -> bool:
        return self.task == "segmentation" and hasattr(self.model, "roi_heads") \
            or self.model.__class__.__name__ == "_Mask2FormerAdapter"

    def _forward_det(self, pil: Image.Image) -> dict:
        from torchvision.transforms.functional import to_tensor
        x = to_tensor(pil).to(self.device)
        out = self.model([x])[0]
        return {k: out[k].cpu() for k in ("boxes", "scores", "labels")}

    def _predict_det(self, pil, score_thr, tile_threshold, tile) -> dict:
        W, H = pil.size
        if max(W, H) > tile_threshold:
            raw = stitch_predictions(self._forward_det, pil, tile=tile,
                                     score_thr=score_thr)
        else:
            raw = self._forward_det(pil)
        keep = raw["scores"] >= score_thr
        boxes = raw["boxes"][keep].numpy()
        scores = raw["scores"][keep].numpy()
        labels = raw["labels"][keep].numpy()
        objs = []
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(v) for v in b)
            area = (x2 - x1) * (y2 - y1)
            objs.append({
                "class": self.class_names[int(l) - 1],
                "confidence": round(float(s), 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1),
                              round(x2, 1), round(y2, 1)],
                "area_px": round(area, 1),
                "area_pct": round(100 * area / (W * H), 4),
                "centroid": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            })
        objs.sort(key=lambda o: -o["confidence"])
        by_class: dict[str, int] = {}
        for o in objs:
            by_class[o["class"]] = by_class.get(o["class"], 0) + 1
        return {"task": "detection", "objects": objs, "counts": by_class}

    def _predict_cls(self, pil) -> dict:
        x = self._tf(pil).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(1)[0].cpu()
        k = min(5, len(self.class_names))
        top = probs.topk(k)
        return {"task": "classification",
                "prediction": self.class_names[int(top.indices[0])],
                "confidence": round(float(top.values[0]), 4),
                "topk": [{"class": self.class_names[int(i)],
                          "prob": round(float(p), 4)}
                         for p, i in zip(top.values, top.indices)]}

    def _predict_semantic(self, pil) -> dict:
        import torch.nn.functional as F
        from torchvision.transforms.functional import normalize, to_tensor
        size = self.cfg.get("img_size", 512)
        x = to_tensor(pil.resize((size, size)))
        x = normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        logits = self.model(x.unsqueeze(0).to(self.device))
        logits = F.interpolate(logits, size=pil.size[::-1], mode="bilinear")
        mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        areas = {}
        total = mask.size
        for cid in np.unique(mask):
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            px = int((mask == cid).sum())
            areas[name] = {"area_px": px, "area_pct": round(100 * px / total, 3)}
        return {"task": "semantic_segmentation", "class_areas": areas,
                "_mask": mask}

    # ------------------------------------------------------------------ #
    def annotate(self, image: str | Path | Image.Image, result: dict,
                 out_path: str | Path | None = None) -> Image.Image:
        pil = (image if isinstance(image, Image.Image)
               else Image.open(image).convert("RGB")).copy()
        if result["task"] == "detection":
            d = ImageDraw.Draw(pil)
            lw = max(2, int(min(pil.size) / 400))
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    max(12, int(min(pil.size) / 70)))
            except OSError:
                font = ImageFont.load_default()
            for o in result["objects"]:
                cid = self.class_names.index(o["class"]) % len(PALETTE)
                color = PALETTE[cid]
                x1, y1, x2, y2 = o["bbox_xyxy"]
                d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
                label = f"{o['class']} {o['confidence']:.2f}"
                tb = d.textbbox((x1, y1), label, font=font)
                d.rectangle([tb[0], tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
                d.text((x1 + 1, y1 - 1), label, fill="white", font=font)
        elif result["task"] == "semantic_segmentation" and "_mask" in result:
            mask = result["_mask"]
            overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for cid in np.unique(mask):
                overlay[mask == cid] = PALETTE[cid % len(PALETTE)]
            pil = Image.blend(pil, Image.fromarray(overlay).resize(pil.size), 0.45)
        elif result["task"] == "classification":
            d = ImageDraw.Draw(pil)
            d.text((8, 8), f"{result['prediction']} ({result['confidence']:.2f})",
                   fill=(255, 40, 40))
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pil.save(out_path)
        return pil

    # ------------------------------------------------------------------ #
    def predict_batch(self, images, out_dir: str | Path,
                      score_thr: float = 0.35, save_annotated: bool = True) -> dict:
        """``images``: list of paths or a directory. Writes results.json,
        results.csv, annotated/ and summary.json under ``out_dir``."""
        out_dir = Path(out_dir)
        (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
        if isinstance(images, (str, Path)) and Path(images).is_dir():
            images = sorted(p for p in Path(images).iterdir()
                            if p.suffix.lower() in
                            {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
        all_results, rows = {}, []
        agg: dict[str, dict] = {}
        for p in images:
            p = Path(p)
            res = self.predict_image(p, score_thr=score_thr)
            if save_annotated:
                self.annotate(p, res, out_dir / "annotated" / p.name)
            res.pop("_mask", None)
            all_results[p.name] = res
            if res["task"] == "detection":
                for o in res["objects"]:
                    rows.append([p.name, o["class"], o["confidence"],
                                 *o["bbox_xyxy"], o["area_px"], o["area_pct"]])
                    a = agg.setdefault(o["class"],
                                       {"count": 0, "conf_sum": 0.0, "area_px": 0.0})
                    a["count"] += 1
                    a["conf_sum"] += o["confidence"]
                    a["area_px"] += o["area_px"]
            elif res["task"] == "classification":
                rows.append([p.name, res["prediction"], res["confidence"],
                             "", "", "", "", "", ""])
        (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
        with (out_dir / "results.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2",
                        "area_px", "area_pct"])
            w.writerows(rows)
        summary = {
            "num_images": len(all_results),
            "per_class": {k: {"count": v["count"],
                              "mean_confidence": round(v["conf_sum"] / v["count"], 4),
                              "total_area_px": round(v["area_px"], 1)}
                          for k, v in sorted(agg.items())},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"results": all_results, "summary": summary, "out_dir": str(out_dir)}
