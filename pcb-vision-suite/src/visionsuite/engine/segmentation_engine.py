"""Segmentation training engine.

Two sub-loops, selected automatically by the zoo bundle's ``kind``:

- **semantic** — logits [B,C,H,W] vs mask [B,H,W]; combined CE + Dice loss;
  logs mIoU/mDice/pixel-acc; final confusion matrix + per-class IoU bars.
- **instance** — torchvision-style API (Mask R-CNN, Mask2Former adapter)
  over COCO instances (rectangular masks are synthesised when only boxes
  exist, as in the PCB dataset); evaluated with COCO box mAP.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from ..data.datasets import (CocoInstanceSegmentation, SemanticSegmentation,
                             split_coco)
from ..data.transforms import SegJointTransform, detection_transforms
from ..engine.common import (Checkpointer, EarlyStopper, pick_device,
                             seed_everything, warmup_cosine)
from ..engine.detection_engine import evaluate_detection
from ..logutils.logger import RunLogger, read_metrics
from ..logutils.plots import (confusion_matrix_plot, per_class_bars,
                              training_curves)
from ..metrics.metrics import SegmentationEvaluator
from ..models.segmentation_zoo import build_segmenter


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    n_cls = logits.shape[1]
    probs = logits.softmax(1)
    onehot = torch.nn.functional.one_hot(
        target.clamp(0, n_cls - 1), n_cls).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def train_segmentation(cfg: dict) -> dict:
    """cfg keys: model, and either ``seg_manifest``+``class_names`` (semantic)
    or ``ann_file`` (instance/COCO). Common: epochs=30, batch_size, lr,
    img_size=512, val_frac=0.2, amp=True, patience=10..."""
    seed_everything(cfg.get("seed", 42))
    device = pick_device(cfg.get("device"))
    logger = RunLogger("segmentation", cfg["model"],
                       cfg.get("run_root", "runs"), cfg.get("run_name"))
    logger.save_config(cfg)

    probe = build_segmenter(cfg["model"], num_classes=2)  # kind lookup only
    if probe.kind == "semantic":
        best = _train_semantic(cfg, device, logger)
    else:
        best = _train_instance(cfg, device, logger)
    training_curves(read_metrics(logger.run_dir), logger.run_dir / "plots")
    (logger.run_dir / "eval" / "best_metrics.json").write_text(
        json.dumps(best, indent=2))
    logger.close()
    return best


# ------------------------------- semantic --------------------------------- #
def _train_semantic(cfg, device, logger) -> dict:
    import numpy as np
    class_names = cfg["class_names"]
    n_cls = len(class_names)
    with open(cfg["seg_manifest"]) as f:
        n_rows = sum(1 for _ in f) - 1
    idx = np.random.RandomState(cfg.get("seed", 42)).permutation(n_rows)
    n_val = max(1, int(n_rows * cfg.get("val_frac", 0.2)))
    va_idx, tr_idx = idx[:n_val].tolist(), idx[n_val:].tolist()

    size = cfg.get("img_size", 512)
    ds_tr = SemanticSegmentation(cfg["seg_manifest"], n_cls,
                                 SegJointTransform(True, size), tr_idx)
    ds_va = SemanticSegmentation(cfg["seg_manifest"], n_cls,
                                 SegJointTransform(False, size), va_idx)
    dl_tr = DataLoader(ds_tr, cfg.get("batch_size", 8), shuffle=True,
                       num_workers=cfg.get("img_workers", 4), pin_memory=True)
    dl_va = DataLoader(ds_va, cfg.get("batch_size", 8), shuffle=False,
                       num_workers=cfg.get("img_workers", 4), pin_memory=True)
    logger.log.info("semantic: train=%d val=%d classes=%d",
                    len(ds_tr), len(ds_va), n_cls)

    model = build_segmenter(cfg["model"], n_cls,
                            cfg.get("pretrained", True)).model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                              weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 30)
    sched = warmup_cosine(optim, epochs, max(1, len(dl_tr)))
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.get("amp", True) and device.type == "cuda")
    ce = torch.nn.CrossEntropyLoss(ignore_index=255)
    ckpt = Checkpointer(logger.run_dir / "checkpoints", "mIoU")
    stopper = EarlyStopper(cfg.get("patience", 10))

    best = {}
    for epoch in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type,
                                enabled=scaler.is_enabled()):
                logits = model(x)
                loss = ce(logits, y) + 0.5 * dice_loss(logits, y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        logger.scalars(epoch, "train", loss=tot / nb,
                       lr=optim.param_groups[0]["lr"])

        model.eval()
        ev = SegmentationEvaluator(class_names)
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss += ce(logits, y).item(); vn += 1
                ev.update(logits, y)
        res = ev.compute()
        logger.scalars(epoch, "val", loss=vloss / max(vn, 1), mIoU=res["mIoU"],
                       mDice=res["mDice"], pixel_accuracy=res["pixel_accuracy"])
        if ckpt.step(res["mIoU"], {"model": model.state_dict(), "epoch": epoch,
                                   "cfg": cfg, "class_names": class_names}):
            best = {"mIoU": res["mIoU"], "mDice": res["mDice"],
                    "pixel_accuracy": res["pixel_accuracy"], "epoch": epoch,
                    "per_class_IoU": res["per_class_IoU"]}
            confusion_matrix_plot(res["confusion_matrix"], class_names,
                                  logger.run_dir / "plots" / "confusion_matrix.png")
            per_class_bars(res["per_class_IoU"],
                           logger.run_dir / "plots" / "per_class_IoU.png", "IoU")
        if stopper.step(res["mIoU"]):
            logger.log.info("early stopping at epoch %d", epoch)
            break
    return best


# ------------------------------- instance --------------------------------- #
def _train_instance(cfg, device, logger) -> dict:
    tr_ids, va_ids = split_coco(cfg["ann_file"], cfg.get("val_frac", 0.2),
                                cfg.get("seed", 42))
    ds_tr = CocoInstanceSegmentation(cfg["ann_file"],
                                     detection_transforms(True), tr_ids)
    ds_va = CocoInstanceSegmentation(cfg["ann_file"],
                                     detection_transforms(False), va_ids)
    class_names = ds_tr.class_names
    dl_tr = DataLoader(ds_tr, cfg.get("batch_size", 2), shuffle=True,
                       num_workers=cfg.get("img_workers", 4),
                       collate_fn=CocoInstanceSegmentation.collate)
    dl_va = DataLoader(ds_va, cfg.get("batch_size", 2), shuffle=False,
                       num_workers=cfg.get("img_workers", 4),
                       collate_fn=CocoInstanceSegmentation.collate)
    logger.log.info("instance: train=%d val=%d classes=%d",
                    len(ds_tr), len(ds_va), len(class_names))

    model = build_segmenter(cfg["model"], len(class_names),
                            cfg.get("pretrained", True)).model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4),
                              weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 25)
    sched = warmup_cosine(optim, epochs, max(1, len(dl_tr)))
    ckpt = Checkpointer(logger.run_dir / "checkpoints", "mAP50")
    stopper = EarlyStopper(cfg.get("patience", 8))

    best = {}
    for epoch in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for images, targets in dl_tr:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optim.step(); sched.step()
            tot += loss.item(); nb += 1
        logger.scalars(epoch, "train", loss=tot / max(nb, 1),
                       lr=optim.param_groups[0]["lr"])

        metrics = evaluate_detection(model, dl_va, class_names, device)
        per_class = metrics.pop("per_class_AP")
        logger.scalars(epoch, "val", **metrics)
        if ckpt.step(metrics["mAP50"], {"model": model.state_dict(),
                                        "epoch": epoch, "cfg": cfg,
                                        "class_names": class_names}):
            best = {**metrics, "epoch": epoch, "per_class_AP": per_class}
            per_class_bars(per_class,
                           logger.run_dir / "plots" / "per_class_AP.png", "AP")
        if stopper.step(metrics["mAP50"]):
            logger.log.info("early stopping at epoch %d", epoch)
            break
    return best
