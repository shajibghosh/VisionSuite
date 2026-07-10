#!/usr/bin/env python
"""One-command preparation of the WACV-2019 PCB Component Detection dataset
(https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection).

The dataset (~47 high-resolution board images, 31 component classes,
~62k annotated instances) must be downloaded manually from the authors'
page (the download link there points to a Google Drive archive). Unzip it
anywhere and pass the folder here. This script then:

1. auto-detects the annotation layout and converts to canonical COCO
2. tiles the 15+ MP boards into 1024px training tiles (20% overlap)
3. crops every component box into a 31-class classification dataset
4. prints ready-to-run training commands for all three tasks

Usage:
    python scripts/prepare_pcb_dataset.py --raw /path/to/pcb_wacv_2019 \
        --out data/pcb
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.data.converters import convert, detect_format
from visionsuite.data.tiling import tile_coco_dataset
from visionsuite.logutils.plots import class_distribution


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="unzipped dataset folder")
    ap.add_argument("--out", default="data/pcb")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.out)

    fmt = detect_format(args.raw)
    print(f"[pcb] detected format: {fmt}")
    if fmt == "unknown":
        sys.exit("Could not recognise the dataset layout. Expected per-board "
                 "folders containing a board image + VOC-style XML.")
    fmt = "pcb_wacv" if fmt in ("pcb_wacv", "voc") else fmt

    det_dir = out / "detection"
    ann = convert(args.raw, det_dir, "detection", fmt)
    data = json.loads(Path(ann).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    counts = Counter(cats[a["category_id"]] for a in data["annotations"])
    print(f"[pcb] images={len(data['images'])} instances={len(data['annotations'])} "
          f"classes={len(cats)}")
    class_distribution(dict(counts), det_dir / "class_distribution.png")

    tiled = tile_coco_dataset(ann, det_dir, tile=args.tile, overlap=args.overlap)
    tdata = json.loads(Path(tiled).read_text())
    print(f"[pcb] tiled -> {len(tdata['images'])} tiles at {args.tile}px")

    cls_dir = out / "classification"
    manifest = convert(args.raw, cls_dir, "classification", fmt)
    print(f"[pcb] classification crops -> {manifest}")

    print(f"""
[pcb] Done. Suggested next steps:

  # detection (any of 14 models — see visionsuite.models.detection_zoo)
  python scripts/train_detection.py --model faster_rcnn_v2 --ann {tiled}
  python scripts/train_detection.py --model yolo11 --ann {tiled} --variant m

  # classification (any of 13 timm models)
  python scripts/train_classification.py --model convnext_v2 --manifest {manifest}

  # instance segmentation (rect masks synthesised from boxes)
  python scripts/train_segmentation.py --model maskrcnn_v2 --ann {tiled}

  # compare everything you've trained
  python scripts/compare_runs.py --task detection

  # web app
  streamlit run webapp/app.py
""")


if __name__ == "__main__":
    main()
