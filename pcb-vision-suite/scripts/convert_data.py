#!/usr/bin/env python
"""Convert any supported annotation format into the suite's canonical format.

Examples
--------
# auto-detect format (VOC / YOLO / COCO / CSV / LabelMe / ImageFolder / PCB)
python scripts/convert_data.py --source data/raw/pcb_wacv_2019 \
    --out data/pcb/detection --task detection

# force a format, then tile the high-res boards for training
python scripts/convert_data.py --source data/raw/pcb_wacv_2019 \
    --out data/pcb/detection --task detection --fmt pcb_wacv \
    --tile 1024 --overlap 0.2

# build the 31-class classification set by cropping every component box
python scripts/convert_data.py --source data/raw/pcb_wacv_2019 \
    --out data/pcb/classification --task classification
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.data.converters import convert, detect_format
from visionsuite.data.tiling import tile_coco_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="annotation dir/file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--task", required=True,
                    choices=["detection", "classification", "segmentation"])
    ap.add_argument("--fmt", default=None,
                    help="force format: coco|voc|yolo|csv|labelme|imagefolder|"
                         "mask_pairs|pcb_wacv (default: auto-detect)")
    ap.add_argument("--tile", type=int, default=0,
                    help="tile size in px for high-res detection images "
                         "(0 = no tiling)")
    ap.add_argument("--overlap", type=float, default=0.2)
    args = ap.parse_args()

    detected = args.fmt or detect_format(args.source)
    print(f"[convert] source={args.source} format={detected} task={args.task}")
    canonical = convert(args.source, args.out, args.task, args.fmt)
    print(f"[convert] canonical annotations -> {canonical}")

    if args.tile and args.task in ("detection", "segmentation") \
            and str(canonical).endswith(".json"):
        tiled = tile_coco_dataset(canonical, Path(args.out),
                                  tile=args.tile, overlap=args.overlap)
        print(f"[convert] tiled dataset      -> {tiled}")
        print("[convert] train on the tiled json; inference auto-stitches.")


if __name__ == "__main__":
    main()
