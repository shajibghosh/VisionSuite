#!/usr/bin/env python
"""Run single-image or batch inference with any trained run.

Examples
--------
# single image (prints JSON with classes, confidences, boxes, areas)
python scripts/infer.py --run runs/detection/faster_rcnn_v2_20260710_101500 \
    --image board_07.jpg --save annotated.jpg

# batch over a folder -> results.json/.csv, annotated images, summary.json
python scripts/infer.py --run runs/detection/faster_rcnn_v2_20260710_101500 \
    --input-dir test_images/ --out predictions/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.inference.predictor import Predictor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--image", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--save", default=None, help="annotated output (single)")
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pred = Predictor.from_run(args.run, device=args.device)
    if args.image:
        res = pred.predict_image(args.image, score_thr=args.score_thr)
        if args.save:
            pred.annotate(args.image, res, args.save)
            print(f"annotated -> {args.save}")
        res.pop("_mask", None)
        print(json.dumps(res, indent=2))
    elif args.input_dir:
        out = pred.predict_batch(args.input_dir, args.out,
                                 score_thr=args.score_thr)
        print(json.dumps(out["summary"], indent=2))
        print(f"full results -> {out['out_dir']}")
    else:
        ap.error("provide --image or --input-dir")


if __name__ == "__main__":
    main()
