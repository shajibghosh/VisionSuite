"""Detection training engine (torchvision + HuggingFace-adapter models).

One loop drives every non-Ultralytics detector because the zoo normalises
them to the torchvision contract:

    train: model(images, targets) -> {loss_name: tensor}
    eval:  model(images)          -> [{'boxes','scores','labels'}, ...]

Per epoch it logs losses + COCO mAP (overall, 50/75, small/medium/large,
per-class) to console/file/TensorBoard/metrics.jsonl, checkpoints best/last
on val mAP50, supports AMP, gradient clipping/accumulation and early
stopping, and finishes by writing all standard plots.
"""
from __future__ import annotations

import json
import math

import torch
from torch.utils.data import DataLoader

from ..data.datasets import CocoDetection, split_coco
from ..data.transforms import detection_transforms
from ..engine.common import (Checkpointer, EarlyStopper, pick_device,
                             seed_everything, warmup_cosine)
from ..logutils.logger import RunLogger
from ..logutils.plots import (class_distribution, per_class_bars,
                              training_curves)
from ..metrics.metrics import DetectionEvaluator
from ..models.detection_zoo import build_detector


def train_detection(cfg: dict) -> dict:
    """cfg keys (with defaults): model, ann_file, epochs=25, batch_size=4,
    lr=5e-4, weight_decay=1e-4, val_frac=0.2, img_workers=4, amp=True,
    grad_clip=10.0, accumulate=1, patience=8, seed=42, pretrained=True,
    variant=None, run_root='runs', run_name=None."""
    seed_everything(cfg.get("seed", 42))
    device = pick_device(cfg.get("device"))
    logger = RunLogger("detection", cfg["model"], cfg.get("run_root", "runs"),
                       cfg.get("run_name"))
    logger.save_config(cfg)

    train_ids, val_ids = split_coco(cfg["ann_file"], cfg.get("val_frac", 0.2),
                                    cfg.get("seed", 42))
    ds_train = CocoDetection(cfg["ann_file"], detection_transforms(True), train_ids)
    ds_val = CocoDetection(cfg["ann_file"], detection_transforms(False), val_ids)
    class_names = ds_train.class_names
    logger.log.info("train=%d val=%d classes=%d device=%s",
                    len(ds_train), len(ds_val), len(class_names), device)

    counts = {}
    for anns in ds_train.anns_by_img.values():
        for a in anns:
            name = ds_train.cats[a["category_id"]]
            counts[name] = counts.get(name, 0) + 1
    class_distribution(counts, logger.run_dir / "plots" / "class_distribution.png")

    dl_train = DataLoader(ds_train, cfg.get("batch_size", 4), shuffle=True,
                          num_workers=cfg.get("img_workers", 4),
                          collate_fn=CocoDetection.collate, pin_memory=True)
    dl_val = DataLoader(ds_val, cfg.get("batch_size", 4), shuffle=False,
                        num_workers=cfg.get("img_workers", 4),
                        collate_fn=CocoDetection.collate, pin_memory=True)

    bundle = build_detector(cfg["model"], len(class_names),
                            cfg.get("pretrained", True), cfg.get("variant"))
    model = bundle.model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.get("lr", 5e-4),
                              weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 25)
    sched = warmup_cosine(optim, epochs, max(1, len(dl_train)))
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.get("amp", True) and device.type == "cuda")
    ckpt = Checkpointer(logger.run_dir / "checkpoints", "mAP50")
    stopper = EarlyStopper(cfg.get("patience", 8))
    accumulate = max(1, cfg.get("accumulate", 1))

    best = {}
    for epoch in range(1, epochs + 1):
        model.train()
        running, nb = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for i, (images, targets) in enumerate(dl_train):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / accumulate
            if not math.isfinite(loss.item()):
                logger.log.warning("non-finite loss %s — skipping batch", loss.item())
                optim.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if (i + 1) % accumulate == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 10.0))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
            running += loss.item() * accumulate
            nb += 1
        logger.scalars(epoch, "train", loss=running / max(nb, 1),
                       lr=optim.param_groups[0]["lr"])

        metrics = evaluate_detection(model, dl_val, class_names, device)
        per_class = metrics.pop("per_class_AP")
        logger.scalars(epoch, "val", **metrics)
        improved = ckpt.step(metrics["mAP50"], {
            "model": model.state_dict(), "epoch": epoch, "cfg": cfg,
            "class_names": class_names, "metrics": metrics})
        if improved:
            best = {**metrics, "epoch": epoch, "per_class_AP": per_class}
            per_class_bars(per_class,
                           logger.run_dir / "plots" / "per_class_AP.png", "AP@50:95")
        if stopper.step(metrics["mAP50"]):
            logger.log.info("early stopping at epoch %d", epoch)
            break

    training_curves(read_rows(logger), logger.run_dir / "plots")
    (logger.run_dir / "eval" / "best_metrics.json").write_text(
        json.dumps(best, indent=2))
    logger.log.info("best: %s", json.dumps({k: v for k, v in best.items()
                                            if k != "per_class_AP"}))
    logger.close()
    return best


@torch.no_grad()
def evaluate_detection(model, loader, class_names, device,
                       score_thr: float = 0.05) -> dict:
    model.eval()
    ev = DetectionEvaluator(class_names)
    for images, targets in loader:
        images = [im.to(device) for im in images]
        outputs = model(images)
        preds = []
        for o in outputs:
            keep = o["scores"] >= score_thr
            preds.append({k: o[k][keep] for k in ("boxes", "scores", "labels")})
        ev.update(preds, targets)
    return ev.compute()


def read_rows(logger: RunLogger):
    from ..logutils.logger import read_metrics
    return read_metrics(logger.run_dir)
