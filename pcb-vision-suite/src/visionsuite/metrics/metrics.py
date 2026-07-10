"""Evaluation metrics for the three tasks.

- Detection / instance segmentation: COCO mAP (via torchmetrics'
  MeanAveragePrecision, which wraps pycocotools conventions), plus
  per-class AP and PR data for plotting.
- Classification: accuracy, balanced accuracy, macro/micro F1, top-5,
  full confusion matrix, per-class precision/recall/F1, PR curves.
- Semantic segmentation: mean IoU, per-class IoU, Dice, pixel accuracy,
  pixel-level confusion matrix.
"""
from __future__ import annotations

import numpy as np
import torch


# ------------------------------- detection -------------------------------- #
class DetectionEvaluator:
    def __init__(self, class_names: list[str]):
        from torchmetrics.detection import MeanAveragePrecision
        self.class_names = class_names
        self.metric = MeanAveragePrecision(class_metrics=True)

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        p = [{"boxes": d["boxes"].cpu(), "scores": d["scores"].cpu(),
              "labels": d["labels"].cpu()} for d in preds]
        t = [{"boxes": d["boxes"].cpu(), "labels": d["labels"].cpu()}
             for d in targets]
        self.metric.update(p, t)

    def compute(self) -> dict:
        m = self.metric.compute()
        out = {
            "mAP": float(m["map"]), "mAP50": float(m["map_50"]),
            "mAP75": float(m["map_75"]), "mAP_small": float(m["map_small"]),
            "mAP_medium": float(m["map_medium"]),
            "mAP_large": float(m["map_large"]),
            "mAR100": float(m["mar_100"]),
        }
        per_class = {}
        if "map_per_class" in m and m["map_per_class"].numel() > 1:
            classes = m.get("classes", torch.arange(len(m["map_per_class"])))
            for cid, ap in zip(classes.tolist(), m["map_per_class"].tolist()):
                idx = int(cid) - 1
                if 0 <= idx < len(self.class_names) and ap >= 0:
                    per_class[self.class_names[idx]] = float(ap)
        out["per_class_AP"] = per_class
        return out

    def reset(self):
        self.metric.reset()


# ----------------------------- classification ----------------------------- #
class ClassificationEvaluator:
    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.logits, self.targets = [], []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.logits.append(logits.detach().float().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        logits = torch.cat(self.logits); targets = torch.cat(self.targets)
        n_cls = len(self.class_names)
        preds = logits.argmax(1)
        acc = (preds == targets).float().mean().item()
        k = min(5, n_cls)
        topk = logits.topk(k, dim=1).indices
        top5 = (topk == targets[:, None]).any(1).float().mean().item()

        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        for t, p in zip(targets.tolist(), preds.tolist()):
            cm[t, p] += 1
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        prec = tp / np.maximum(tp + fp, 1e-9)
        rec = tp / np.maximum(tp + fn, 1e-9)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        support = cm.sum(1)
        present = support > 0
        balanced_acc = rec[present].mean() if present.any() else 0.0

        probs = torch.softmax(logits, 1).numpy()
        pr = {}
        for c in range(n_cls):
            if support[c] == 0:
                continue
            p_arr, r_arr = _pr_curve(probs[:, c], (targets.numpy() == c))
            pr[self.class_names[c]] = (p_arr, r_arr)

        return {
            "accuracy": acc, "top5_accuracy": top5,
            "balanced_accuracy": float(balanced_acc),
            "macro_f1": float(f1[present].mean() if present.any() else 0),
            "micro_f1": acc,
            "confusion_matrix": cm,
            "per_class": {
                self.class_names[c]: {"precision": float(prec[c]),
                                      "recall": float(rec[c]),
                                      "f1": float(f1[c]),
                                      "support": int(support[c])}
                for c in range(n_cls)},
            "pr_curves": pr,
        }


def _pr_curve(scores: np.ndarray, positives: np.ndarray):
    order = np.argsort(-scores)
    pos = positives[order]
    tp = np.cumsum(pos)
    fp = np.cumsum(~pos)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / max(pos.sum(), 1e-9)
    return prec, rec


# ------------------------------ segmentation ------------------------------ #
class SegmentationEvaluator:
    def __init__(self, class_names: list[str], ignore_index: int = 255):
        self.class_names = class_names
        self.n = len(class_names)
        self.ignore = ignore_index
        self.cm = np.zeros((self.n, self.n), dtype=np.int64)

    def reset(self):
        self.cm[:] = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = logits.argmax(1).flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        keep = (t != self.ignore) & (t < self.n)
        idx = self.n * t[keep].astype(np.int64) + pred[keep]
        self.cm += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self) -> dict:
        cm = self.cm.astype(float)
        tp = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - tp
        iou = tp / np.maximum(union, 1e-9)
        dice = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        present = cm.sum(1) > 0
        return {
            "mIoU": float(iou[present].mean() if present.any() else 0),
            "mDice": float(dice[present].mean() if present.any() else 0),
            "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1e-9)),
            "per_class_IoU": {self.class_names[i]: float(iou[i])
                              for i in range(self.n) if present[i]},
            "confusion_matrix": self.cm.copy(),
        }
