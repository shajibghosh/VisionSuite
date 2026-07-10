# VisionSuite — PCB Component Detection, Classification & Segmentation

A unified, scalable fine-tuning suite covering **14 detection**, **13
classification**, and **15 segmentation** SOTA architectures, with universal
annotation-format ingestion, first-class logging/plots/TensorBoard, run
comparison, batch/single inference reporting, and a Streamlit web app.

Primary use case: the **WACV-2019 PCB Component Detection dataset**
(<https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection>) —
47 high-resolution board images, **31 component classes**, ~62k annotated
instances with a heavily skewed class distribution. The suite ships specific
machinery for this dataset: a dedicated converter, **image tiling** for the
15+ MP boards, box→crop generation for component classification, and
class-imbalance handling (weighted sampling / weighted loss / balanced
accuracy monitoring).

---

## 1. Install

```bash
git clone <this repo> && cd pcb-vision-suite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.2. GPU strongly recommended for training; CPU is
fine for inference and the web app.

## 2. Quickstart on the PCB dataset

Download the dataset archive from the authors' page (link on the site above),
unzip it, then run the one-command preparer:

```bash
python scripts/prepare_pcb_dataset.py --raw /path/to/pcb_wacv_2019 --out data/pcb
```

This auto-detects the annotation layout, converts to canonical COCO, tiles
each board into 1024 px training tiles with 20 % overlap, crops all ~62k
component boxes into a 31-class classification dataset, and prints
ready-to-run training commands. Then, for example:

```bash
# detection — pick any of 14 models
python scripts/train_detection.py --model faster_rcnn_v2 \
    --ann data/pcb/detection/annotations_tiled.json --epochs 25
python scripts/train_detection.py --model yolo11 --variant m \
    --ann data/pcb/detection/annotations_tiled.json --epochs 100

# classification — pick any of 13 models
python scripts/train_classification.py --model convnext_v2 \
    --manifest data/pcb/classification/manifest.csv

# instance segmentation (rect masks synthesized from boxes)
python scripts/train_segmentation.py --model maskrcnn_v2 \
    --ann data/pcb/detection/annotations_tiled.json

# leaderboard + comparison plots across everything you trained
python scripts/compare_runs.py --task detection

# web app: dashboards, comparisons, single & batch inference
streamlit run webapp/app.py
```

## 3. Model zoos

List keys at any time with `--list` on each training script.

**Detection (14)** — `faster_rcnn`, `faster_rcnn_v2`, `retinanet`,
`retinanet_v2`, `fcos`, `ssd300`, `ssdlite` (torchvision); `detr`,
`deformable_detr`, `conditional_detr`, `yolos` (HuggingFace); `yolov8`,
`yolo11`, `rtdetr` (Ultralytics, `--variant n/s/m/l/x`).

**Classification (13)** — `resnet50`, `convnext`, `convnext_v2`, `vit`,
`deit3`, `swin`, `swin_v2`, `efficientnet` (EfficientNetV2-S), `maxvit`,
`regnety`, `mobilenetv3`, `densenet`, `eva02` — all via timm; any other timm
checkpoint also works: `--model timm/<name>`.

**Segmentation (15)** — semantic: `unet`, `unetplusplus`, `deeplabv3plus`,
`fpn`, `pspnet`, `manet`, `linknet`, `pan` (SMP); `deeplabv3`, `fcn`,
`lraspp` (torchvision); `segformer` (HF). Instance: `maskrcnn`,
`maskrcnn_v2` (torchvision), `mask2former` (HF).

All non-Ultralytics detectors are normalized to the torchvision API
(`model(images, targets) → loss dict`; `model(images) → detections`) via thin
adapters, so **one engine trains them all**. Ultralytics models train through
their own optimized engine and their metrics are harvested into the same run
format, so comparison and the web app treat every run identically.

## 4. Any data format in → canonical format out

`scripts/convert_data.py` auto-detects and converts:

| input format | detection | classification | segmentation |
|---|---|---|---|
| COCO JSON | ✅ passthrough | ✅ via crops | ✅ (polygons/RLE) |
| Pascal VOC XML | ✅ | ✅ via crops | rect-mask instance |
| YOLO txt (+classes.txt) | ✅ | ✅ via crops | rect-mask instance |
| CSV (flexible headers) | ✅ | ✅ | — |
| LabelMe JSON | ✅ | ✅ via crops | ✅ polygons |
| ImageFolder (`root/class/img`) | — | ✅ | — |
| `images/` + `masks/` PNG pairs | — | — | ✅ semantic |
| PCB WACV-2019 layout | ✅ | ✅ via crops | rect-mask instance |

Canonical formats: detection/instance-seg → **COCO JSON**; classification →
**manifest.csv + classes.txt**; semantic seg → **seg_manifest.csv** of
image/mask pairs. CSV headers are alias-mapped (`filename/x_min/x1/left`
etc. all understood). Force a format with `--fmt` if auto-detection guesses
wrong.

**High-resolution tiling**: add `--tile 1024 --overlap 0.2` to slice big
images for training (`annotations_tiled.json`). At inference, the
`Predictor` automatically tiles any image larger than 2048 px and stitches
detections back with global NMS — so you predict on full boards directly.

## 5. Logging, plots & reports (every run, every model)

Each run creates a self-describing directory:

```
runs/<task>/<model>_<timestamp>/
├── config.json           # full resolved config (reproducibility)
├── train.log             # console mirror
├── tensorboard/          # tensorboard --logdir runs
├── metrics.jsonl         # one JSON line per epoch (machine-readable)
├── checkpoints/          # best.pt + last.pt (resume-ready)
├── plots/                # loss/metric curves, confusion matrix, PR curves,
│                         # per-class AP/IoU/F1 bars, class distribution
└── eval/                 # best_metrics.json, classification_report.json
```

Logged metrics: detection — COCO mAP/mAP50/mAP75, small/medium/large mAP,
mAR, per-class AP; classification — accuracy, **balanced accuracy** (the
right headline metric for skewed PCB classes), macro-F1, top-5, per-class
precision/recall/F1, confusion matrix, PR curves; segmentation — mIoU,
mDice, pixel accuracy, per-class IoU, pixel confusion matrix.

Training features: AMP mixed precision, warmup + cosine LR, gradient
clipping & accumulation, early stopping, weighted sampling or weighted CE
for imbalance, mixup + label smoothing, layer-wise LR decay for ViTs,
deterministic seeding.

## 6. Inference & reporting

```bash
# single image → JSON with class, confidence, bbox, pixel area, % area, centroid
python scripts/infer.py --run runs/detection/<run> --image board.jpg --save out.jpg

# batch → results.json, results.csv, annotated/, summary.json
python scripts/infer.py --run runs/detection/<run> --input-dir test_images/ --out preds/
```

Batch `summary.json` aggregates per class: object count, mean confidence and
total pixel area — e.g. "resistor: 412 detections, mean conf 0.91,
total area 1.2 Mpx" across a folder of boards.

## 7. Web app

```bash
streamlit run webapp/app.py -- --runs-root runs
```

- **Dashboard** — pick any run: config, interactive metric curves, all saved
  plots, best-checkpoint metrics, per-class tables.
- **Compare models** — sortable leaderboard, downloadable CSV, comparison
  bar charts and overlaid validation curves.
- **Single inference** — upload an image, choose any trained run, adjust the
  score threshold, view side-by-side annotated output plus a per-object
  table (class / confidence / box / area px / area %) and per-class counts.
- **Batch inference** — upload many images or point at a folder; annotated
  gallery, per-class summary, downloads for results.csv / results.json /
  annotated.zip.

## 8. Extending

- **New detector**: add a `@register("name")` builder in
  `src/visionsuite/models/detection_zoo.py` returning a `ModelBundle` whose
  model follows the torchvision contract (wrap it if it doesn't). Done —
  training, eval, inference, comparison and the web app pick it up.
- **New annotation format**: add a `read_<fmt>()` returning `DatasetIR` in
  `converters/converters.py` and register it in `READERS` (+ a sniffing rule
  in `base.detect_format`).
- **New metric/plot**: emit it via `logger.scalars(...)` — it automatically
  appears in TensorBoard, metrics.jsonl, curve plots, the leaderboard and
  the web app.

## 9. Repository layout

See `docs/CODE_GUIDE.md` for a file-by-file explanation of every module,
and `docs/README.docx` for the Word version of the full documentation.

```
configs/            example YAML configs (detection/classification/segmentation)
scripts/            CLI entrypoints (convert, prepare PCB, train×3, infer, compare)
src/visionsuite/
  data/converters/  format sniffing + all converters → canonical formats
  data/datasets.py  canonical-format PyTorch datasets
  data/transforms.py  task-appropriate augmentation pipelines
  data/tiling.py    high-res tiling (train) + prediction stitching (infer)
  models/           three model-zoo registries (42 models total)
  engine/           shared trainer utilities + one engine per task
  metrics/          COCO mAP / classification / segmentation evaluators
  logutils/         RunLogger (console+file+TB+jsonl) and all plotting
  inference/        Predictor: single/batch, tiling, annotation, reports
  compare.py        cross-run leaderboards and comparison plots
webapp/app.py       Streamlit app (dashboard / compare / single / batch)
tests/              dependency-light smoke tests (pytest)
```

## 10. Tests

```bash
python -m pytest tests/ -q
```

Covers format conversion, tiling geometry/box remapping, and the
classification & segmentation evaluators without requiring GPUs or model
downloads.