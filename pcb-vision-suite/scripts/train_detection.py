#!/usr/bin/env python
"""Fine-tune any of the 14 registered detectors on a canonical COCO dataset.

Examples
--------
python scripts/train_detection.py --model faster_rcnn_v2 \
    --ann data/pcb/detection/annotations_tiled.json --epochs 25

python scripts/train_detection.py --model yolo11 --variant m \
    --ann data/pcb/detection/annotations_tiled.json --epochs 100

python scripts/train_detection.py --list          # show all model keys
python scripts/train_detection.py --config configs/detection_pcb.yaml

Config-file values are overridden by CLI flags. Ultralytics models
(yolov8/yolo11/rtdetr) are trained through the Ultralytics engine after an
automatic COCO->YOLO export; their metrics are harvested into the same
run-directory format so comparison and the web app treat all runs equally.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.models.detection_zoo import REGISTRY, build_detector, list_models


def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--config", default=None, help="YAML config file")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ann", dest="ann_file", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--img-size", type=int, default=None,
                    help="ultralytics only (torchvision detectors resize internally)")
    ap.add_argument("--variant", default=None, help="e.g. n/s/m/l/x for YOLO")
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        print("\n".join(list_models()))
        return

    cfg = load_config(args.config) if args.config else {}
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "list"):
            cfg[k] = v
    if "model" not in cfg or "ann_file" not in cfg:
        ap.error("--model and --ann are required (or provide them in --config)")

    if cfg["model"] not in REGISTRY:
        ap.error(f"unknown model {cfg['model']}; use --list")

    probe = build_detector(cfg["model"], num_classes=1, pretrained=False) \
        if cfg["model"] in ("yolov8", "yolo11", "rtdetr") else None
    if probe is not None and probe.framework == "ultralytics":
        train_ultralytics(cfg)
    else:
        from visionsuite.engine.detection_engine import train_detection
        best = train_detection(cfg)
        print(json.dumps({k: v for k, v in best.items() if k != "per_class_AP"},
                         indent=2))


def train_ultralytics(cfg: dict) -> None:
    """Train YOLO/RT-DETR via Ultralytics, then harvest results into the
    suite's run-directory contract (config.json / metrics.jsonl /
    eval/best_metrics.json / plots/) so all tooling downstream just works."""
    from ultralytics import RTDETR, YOLO

    from visionsuite.data.converters import coco_to_yolo
    from visionsuite.logutils.logger import RunLogger

    logger = RunLogger("detection", cfg["model"], cfg.get("run_root", "runs"),
                       cfg.get("run_name"))
    logger.save_config(cfg)
    yolo_dir = Path(cfg["ann_file"]).parent / "yolo_export"
    data_yaml = coco_to_yolo(cfg["ann_file"], yolo_dir)
    logger.log.info("exported COCO -> YOLO at %s", data_yaml)

    from visionsuite.models.detection_zoo import ULTRA_DEFAULT_WEIGHTS
    v = cfg.get("variant") or ("s" if cfg["model"] != "rtdetr" else "")
    weights = ULTRA_DEFAULT_WEIGHTS[cfg["model"]].format(v=v)
    model = RTDETR(weights) if cfg["model"] == "rtdetr" else YOLO(weights)

    results = model.train(
        data=str(data_yaml),
        epochs=cfg.get("epochs", 100),
        imgsz=cfg.get("img_size", 1024),
        batch=cfg.get("batch_size", 8),
        lr0=cfg.get("lr", 0.01),
        seed=cfg.get("seed", 42),
        device=cfg.get("device"),
        project=str(logger.run_dir), name="ultralytics",
        exist_ok=True, plots=True,
    )

    # Harvest per-epoch CSV into metrics.jsonl
    import csv as _csv
    res_csv = logger.run_dir / "ultralytics" / "results.csv"
    if res_csv.exists():
        with res_csv.open() as f:
            for row in _csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                ep = int(float(row.get("epoch", 0)))
                logger.scalars(ep, "train",
                               loss=float(row.get("train/box_loss", 0)) +
                               float(row.get("train/cls_loss", 0)))
                logger.scalars(ep, "val",
                               mAP50=float(row.get("metrics/mAP50(B)", 0)),
                               mAP=float(row.get("metrics/mAP50-95(B)", 0)),
                               precision=float(row.get("metrics/precision(B)", 0)),
                               recall=float(row.get("metrics/recall(B)", 0)))
    box = getattr(results, "box", None)
    best = {"mAP50": float(getattr(box, "map50", 0.0)),
            "mAP": float(getattr(box, "map", 0.0)),
            "mAP75": float(getattr(box, "map75", 0.0))}
    (logger.run_dir / "eval" / "best_metrics.json").write_text(
        json.dumps(best, indent=2))
    # expose weights where Predictor-style tooling expects them
    src = logger.run_dir / "ultralytics" / "weights" / "best.pt"
    if src.exists():
        (logger.run_dir / "checkpoints").mkdir(exist_ok=True)
        (logger.run_dir / "checkpoints" / "best_ultralytics.pt").write_bytes(
            src.read_bytes())
    logger.log.info("done: %s", json.dumps(best))
    logger.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[train_detection] finished in {time.time() - t0:.1f}s")
