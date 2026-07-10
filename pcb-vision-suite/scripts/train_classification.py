#!/usr/bin/env python
"""Fine-tune any of the 13 registered classifiers on a manifest dataset.

Examples
--------
python scripts/train_classification.py --model convnext_v2 \
    --manifest data/pcb/classification/manifest.csv --epochs 30

python scripts/train_classification.py --model vit --freeze-backbone \
    --manifest data/pcb/classification/manifest.csv     # linear probe

python scripts/train_classification.py --list
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.models.classification_zoo import list_models


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--balance", choices=["sampler", "loss", "none"], default=None)
    ap.add_argument("--freeze-backbone", action="store_true", default=None)
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
    if cfg.get("balance") == "none":
        cfg["balance"] = None
    if "model" not in cfg or "manifest" not in cfg:
        ap.error("--model and --manifest are required")

    from visionsuite.engine.classification_engine import train_classification
    best = train_classification(cfg)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
