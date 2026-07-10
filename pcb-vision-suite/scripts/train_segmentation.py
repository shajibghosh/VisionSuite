#!/usr/bin/env python
"""Fine-tune any of the 15 registered segmentation models.

Semantic models (unet, unetplusplus, deeplabv3plus, fpn, pspnet, manet,
linknet, pan, deeplabv3, fcn, lraspp, segformer) train on image/mask pairs:

    python scripts/train_segmentation.py --model segformer \
        --seg-manifest data/xyz/seg_manifest.csv \
        --classes background pad trace via

Instance models (maskrcnn, maskrcnn_v2, mask2former) train on COCO
annotations (rectangular masks are synthesised when only boxes exist —
which is exactly the PCB dataset case):

    python scripts/train_segmentation.py --model maskrcnn_v2 \
        --ann data/pcb/detection/annotations_tiled.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.models.segmentation_zoo import list_models


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ann", dest="ann_file", default=None)
    ap.add_argument("--seg-manifest", dest="seg_manifest", default=None)
    ap.add_argument("--classes", nargs="*", dest="class_names", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        print("\n".join(list_models()))
        return

    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "list"):
            cfg[k] = v
    if "model" not in cfg:
        ap.error("--model is required")
    if not cfg.get("ann_file") and not cfg.get("seg_manifest"):
        ap.error("provide --ann (instance) or --seg-manifest + --classes (semantic)")

    from visionsuite.engine.segmentation_engine import train_segmentation
    best = train_segmentation(cfg)
    print(json.dumps({k: v for k, v in best.items()
                      if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
