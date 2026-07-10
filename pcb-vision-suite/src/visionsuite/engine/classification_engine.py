"""Classification training engine for all timm zoo models.

Highlights for the PCB use case (31 component classes, heavily skewed):

- WeightedRandomSampler *or* class-weighted cross-entropy (config toggle)
- label smoothing + optional mixup
- layer-wise LR decay for transformer backbones
- logs accuracy / balanced accuracy / macro-F1 / top-5 per epoch to
  console, TensorBoard and metrics.jsonl
- final artifacts: confusion matrix, PR curves, per-class F1 bars,
  classification_report.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.datasets import ManifestClassification
from ..data.transforms import classification_transforms
from ..engine.common import (Checkpointer, EarlyStopper, pick_device,
                             seed_everything, warmup_cosine)
from ..logutils.logger import RunLogger, read_metrics
from ..logutils.plots import (confusion_matrix_plot, per_class_bars,
                              pr_curves, training_curves)
from ..metrics.metrics import ClassificationEvaluator
from ..models.classification_zoo import (build_classifier, param_groups_lrd,
                                         resolve_input_size, resolve_norm)


def _split(n, val_frac, seed):
    idx = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def train_classification(cfg: dict) -> dict:
    """cfg keys: model, manifest, epochs=30, batch_size=64, lr=3e-4,
    weight_decay=0.05, val_frac=0.2, img_size=None, freeze_backbone=False,
    balance='sampler'|'loss'|None, mixup=0.2, label_smoothing=0.1, ..."""
    seed_everything(cfg.get("seed", 42))
    device = pick_device(cfg.get("device"))
    logger = RunLogger("classification", cfg["model"],
                       cfg.get("run_root", "runs"), cfg.get("run_name"))
    logger.save_config(cfg)

    probe = ManifestClassification(cfg["manifest"])
    tr_idx, va_idx = _split(len(probe), cfg.get("val_frac", 0.2),
                            cfg.get("seed", 42))
    class_names = probe.class_names
    model = build_classifier(cfg["model"], len(class_names),
                             cfg.get("pretrained", True),
                             cfg.get("freeze_backbone", False)).to(device)
    img_size = cfg.get("img_size") or resolve_input_size(model)
    mean, std = resolve_norm(model)

    ds_train = ManifestClassification(
        cfg["manifest"], classification_transforms(True, img_size, mean, std), tr_idx)
    ds_val = ManifestClassification(
        cfg["manifest"], classification_transforms(False, img_size, mean, std), va_idx)
    logger.log.info("train=%d val=%d classes=%d img=%d device=%s",
                    len(ds_train), len(ds_val), len(class_names), img_size, device)

    balance = cfg.get("balance", "sampler")
    sampler = None
    if balance == "sampler":
        labels = ds_train.labels()
        counts = np.bincount(labels, minlength=len(class_names))
        w = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler([w[l] for l in labels],
                                        num_samples=len(labels), replacement=True)
    dl_train = DataLoader(ds_train, cfg.get("batch_size", 64),
                          shuffle=sampler is None, sampler=sampler,
                          num_workers=cfg.get("img_workers", 4), pin_memory=True,
                          drop_last=True)
    dl_val = DataLoader(ds_val, cfg.get("batch_size", 64), shuffle=False,
                        num_workers=cfg.get("img_workers", 4), pin_memory=True)

    class_w = ds_train.class_weights().to(device) if balance == "loss" else None
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_w, label_smoothing=cfg.get("label_smoothing", 0.1))

    groups = param_groups_lrd(model, cfg.get("lr", 3e-4),
                              cfg.get("weight_decay", 0.05),
                              cfg.get("layer_decay", 0.75))
    optim = torch.optim.AdamW(groups)
    epochs = cfg.get("epochs", 30)
    sched = warmup_cosine(optim, epochs, max(1, len(dl_train)),
                          cfg.get("warmup_epochs", 2))
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.get("amp", True) and device.type == "cuda")
    ckpt = Checkpointer(logger.run_dir / "checkpoints", "balanced_accuracy")
    stopper = EarlyStopper(cfg.get("patience", 10))
    mixup_a = cfg.get("mixup", 0.2)

    best = {}
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = run_acc = nb = 0
        for x, y in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lam, y2 = 1.0, y
            if mixup_a > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(mixup_a, mixup_a))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
                y2 = y[perm]
            with torch.autocast(device_type=device.type,
                                enabled=scaler.is_enabled()):
                out = model(x)
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y2)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            run_acc += (out.argmax(1) == y).float().mean().item()
            nb += 1
        logger.scalars(epoch, "train", loss=run_loss / nb, accuracy=run_acc / nb,
                       lr=optim.param_groups[0]["lr"])

        res = evaluate_classification(model, dl_val, class_names, device)
        scalars = {k: res[k] for k in ("accuracy", "balanced_accuracy",
                                       "macro_f1", "top5_accuracy")}
        val_loss = res.pop("_val_loss", None)
        if val_loss is not None:
            scalars["loss"] = val_loss
        logger.scalars(epoch, "val", **scalars)
        if ckpt.step(res["balanced_accuracy"], {
                "model": model.state_dict(), "epoch": epoch, "cfg": cfg,
                "class_names": class_names, "metrics": scalars}):
            best = {**scalars, "epoch": epoch}
            _final_plots(res, class_names, logger)
        if stopper.step(res["balanced_accuracy"]):
            logger.log.info("early stopping at epoch %d", epoch)
            break

    training_curves(read_metrics(logger.run_dir), logger.run_dir / "plots")
    (logger.run_dir / "eval" / "best_metrics.json").write_text(
        json.dumps(best, indent=2))
    logger.close()
    return best


@torch.no_grad()
def evaluate_classification(model, loader, class_names, device) -> dict:
    model.eval()
    ev = ClassificationEvaluator(class_names)
    tot_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot_loss += Fn.cross_entropy(out, y, reduction="sum").item()
        n += y.numel()
        ev.update(out, y)
    res = ev.compute()
    res["_val_loss"] = tot_loss / max(n, 1)
    return res


def _final_plots(res: dict, class_names, logger: RunLogger) -> None:
    plots = logger.run_dir / "plots"
    confusion_matrix_plot(res["confusion_matrix"], class_names,
                          plots / "confusion_matrix.png")
    if res["pr_curves"]:
        precs = {k: v[0] for k, v in res["pr_curves"].items()}
        recs = {k: v[1] for k, v in res["pr_curves"].items()}
        pr_curves(precs, recs, plots / "pr_curves.png")
    per_class_bars({k: v["f1"] for k, v in res["per_class"].items()},
                   plots / "per_class_f1.png", "F1")
    (logger.run_dir / "eval" / "classification_report.json").write_text(
        json.dumps({"per_class": res["per_class"],
                    "confusion_matrix": res["confusion_matrix"].tolist()},
                   indent=2))
