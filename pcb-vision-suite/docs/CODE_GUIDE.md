# Code Guide — every file explained

This document walks through each source file: what it does, the key design
decisions, and how the pieces connect. Read alongside `README.md`.

The suite's architecture in one sentence: **converters normalize any data
into three canonical formats; model zoos normalize 42 architectures into
three call contracts; so three engines, one predictor, one logger, one
comparison module and one web app cover everything.**

---

## `src/visionsuite/data/converters/base.py`

Defines the **intermediate representation (IR)** shared by all converters and
the **format sniffer**.

- `Instance` / `ImageRecord` / `DatasetIR` — a tiny dataclass model of "a
  dataset": images with sizes, per-image object instances (category, xywh
  box, optional polygon), an optional classification label, an optional
  semantic mask path. Every reader produces this; every writer consumes it.
  This is what makes N formats × 3 tasks tractable: N readers + 3 writers
  instead of N×3 converters.
- `ir_to_coco()` — serializes the IR to canonical COCO (contiguous category
  ids, computed areas, optional segmentation polygons).
- `ir_to_classification_manifest()` — writes `manifest.csv` + `classes.txt`.
- `detect_format()` — layered heuristics: COCO JSON key check, LabelMe
  `shapes` key, VOC directory conventions and `<annotation>` XML roots, YOLO
  numeric-txt shape, ImageFolder class-subdirectory shape, `images/`+`masks/`
  pairs, and a specific detector for the WACV PCB layout (per-board folders
  each holding a board image + XML). Anything ambiguous can be forced via
  `--fmt`.

## `src/visionsuite/data/converters/converters.py`

One `read_<format>()` per supported input, each returning a `DatasetIR`:

- `read_coco` — parses COCO, resolving image paths relative to the JSON.
- `read_voc` — walks all `*.xml`, tolerating both classic VOC layout and
  loose "XML-next-to-image" layouts; reads sizes from XML or the image.
- `read_yolo` — resolves class names from `classes.txt`/`obj.names`,
  denormalizes `cx cy w h` boxes, supports both `label-next-to-image` and
  the `images/`↔`labels/` twin-tree convention.
- `read_csv_annotations` — header alias mapping (`CSV_COL_ALIASES`) so
  `filename/x_min/left/x1` etc. all work; degrades gracefully to a
  classification CSV when box columns are absent.
- `read_labelme` — polygons become both a bbox and a COCO segmentation.
- `read_imagefolder` — `root/class_x/img.jpg` for classification.
- `read_mask_pairs` — matches `images/*` to `masks/*` by stem.
- `read_pcb_wacv` — VOC reader + class-name normalisation (the PCB XMLs vary
  in casing/whitespace across boards).

Cross-task/format bridges:

- `convert()` — the single entrypoint: sniff (or accept) the format, read to
  IR, write the canonical output for the requested task. Notably, asking for
  `classification` from a detection-style source routes through…
- `crops_from_detections()` — crops every annotated box into
  `crops/<class>/`, producing the 31-class PCB component classification set
  from the detection annotations. Boxes under 8 px are skipped.
- `coco_to_yolo()` — exports canonical COCO to an Ultralytics dataset
  (symlinked images, generated label txts, `data.yaml`) so YOLO/RT-DETR can
  train from the exact same canonical data as everything else.

## `src/visionsuite/data/datasets.py`

PyTorch `Dataset`s over the canonical formats only:

- `CocoDetection` — returns `(image, target)` in the torchvision convention:
  `boxes` (xyxy), `labels` **1-based** (0 = background), `image_id`, `area`,
  `iscrowd`. Category ids are remapped to contiguous. Degenerate boxes
  (≤1 px) are dropped. `collate` provides the list-style batch detectors
  expect. `split_coco()` gives a deterministic image-level train/val split.
- `ManifestClassification` — manifest rows + `classes.txt`;
  `class_weights()` computes inverse-frequency weights for the skewed PCB
  distribution; `labels()` feeds the weighted sampler.
- `SemanticSegmentation` — image/mask pair loader.
- `CocoInstanceSegmentation` — extends `CocoDetection` with a `masks`
  tensor: real polygon masks when the COCO file has them, otherwise
  **rectangular masks synthesized from boxes** — this is what lets Mask
  R-CNN / Mask2Former fine-tune on the box-only PCB dataset.

`Image.MAX_IMAGE_PIXELS = None` disables PIL's decompression-bomb guard,
required for 15+ MP board photos.

## `src/visionsuite/data/transforms.py`

- Detection: joint `(image, target)` transforms — photometric jitter (safe
  for boxes; matters because PCB boards differ in lighting/exposure) and a
  box-aware horizontal flip. Geometric resizing is left to the detectors'
  internal `GeneralizedRCNNTransform`/processor.
- Classification: ImageNet-style train/eval pipelines with vertical flips
  (components appear in any orientation) and RandomErasing.
- Segmentation: `SegJointTransform` applies identical resize/flip to image
  and mask (nearest-neighbour for masks) and normaliszes the image.

## `src/visionsuite/data/tiling.py`

The PCB-critical module.

- `_tile_grid()` — overlapping tile origins guaranteed to cover the image
  (adds edge-aligned tiles).
- `tile_coco_dataset()` — pre-slices every image to `tiles/`, clips boxes to
  each tile, drops boxes whose visible fraction < `min_visibility` (40 %),
  and writes `annotations_tiled.json`. Training then needs no special code.
- `stitch_predictions()` — inference-time inverse: run any per-tile predict
  function, shift boxes back to board coordinates, and merge duplicates from
  overlap regions with class-aware global NMS (`batched_nms`).

## `src/visionsuite/models/detection_zoo.py`

Registry of 14 detectors returning a `ModelBundle(name, framework, model)`.

- **torchvision (7)** — built with pretrained weights, then the
  classification head is swapped for `num_classes+1` outputs. Each family
  needs a different head-surgery function (`_fix_frcnn`, `_fix_retina`,
  `_fix_fcos`, `_fix_ssd`, `_fix_ssdlite`) because their heads differ.
- **HuggingFace (4)** — `HFDetrAdapter` wraps any
  `AutoModelForObjectDetection` (DETR, Deformable DETR, Conditional DETR,
  YOLOS) behind the torchvision contract: converts targets to normalized
  cxcywh + 0-based labels for training, and post-processes eval outputs
  back to absolute xyxy + 1-based labels. This is the trick that lets one
  engine train both families.
- **Ultralytics (3)** — declared with `framework="ultralytics"`; the
  training script routes these to Ultralytics' own trainer (best recipes for
  YOLO) and harvests results back.

Adding a model is a `@register("key")` function — nothing else changes.

## `src/visionsuite/models/classification_zoo.py`

A name→timm-checkpoint table (13 entries) plus:

- `build_classifier()` — timm creation with head replacement;
  `freeze_backbone=True` gives a linear-probe baseline.
- `resolve_input_size()` / `resolve_norm()` — read each checkpoint's
  pretrained config so every model automatically trains at its native
  resolution and normalisation (e.g. EVA-02 at 336, SwinV2 at 256).
- `param_groups_lrd()` — layer-wise LR decay for ViT-style backbones
  (earlier blocks get exponentially smaller LRs), with a decay/no-decay
  two-group fallback for CNNs.

## `src/visionsuite/models/segmentation_zoo.py`

15 models in one registry, tagged `kind="semantic"` or `"instance"`:

- **SMP (8)** — U-Net, U-Net++, DeepLabV3+, FPN, PSPNet, MA-Net, LinkNet,
  PAN, each with a configurable ImageNet-pretrained encoder.
- **torchvision (3)** — DeepLabV3/FCN/LR-ASPP, wrapped by
  `_TVSemanticAdapter` to return plain logits instead of `{'out': …}`, with
  proper head replacement per architecture.
- **HF (2)** — SegFormer (adapter upsamples its ¼-resolution logits) and
  Mask2Former (adapter translates between torchvision-style instance targets
  and HF's mask/class-label API, and converts its universal-segmentation
  output back to per-instance masks/boxes/scores).
- **torchvision instance (2)** — Mask R-CNN v1/v2 with box+mask head swaps.

The uniform contract — semantic: `forward(x)→[B,C,H,W]`; instance:
torchvision detector API — is what keeps the engine count at one per kind.

## `src/visionsuite/engine/common.py`

Shared machinery: seeding, device pick (CUDA→MPS→CPU), warmup+cosine LR
lambda, `Checkpointer` (last.pt every epoch, best.pt on monitored-metric
improvement), `EarlyStopper`.

## `src/visionsuite/engine/detection_engine.py`

The single loop for all torchvision + HF detectors: AMP autocast +
GradScaler, gradient accumulation and clipping, non-finite-loss batch
skipping (DETR warmup can spike), per-epoch COCO evaluation
(`DetectionEvaluator`), `logger.scalars()` for every number (console + file
+ TensorBoard + metrics.jsonl in one call), best/last checkpointing on val
mAP50, early stopping, and end-of-run artifacts: class-distribution plot,
per-class AP bars, all metric curves, `eval/best_metrics.json`.

## `src/visionsuite/engine/classification_engine.py`

Imbalance-aware classification training: `WeightedRandomSampler` or
class-weighted CE (config `balance:`), mixup + label smoothing, layer-wise
LR decay, native-resolution transforms per model. Checkpoints on **balanced
accuracy** (headline accuracy is misleading when one class is 40 % of the
data). Final artifacts: confusion matrix, PR curves, per-class F1 bars,
`classification_report.json`.

## `src/visionsuite/engine/segmentation_engine.py`

Dispatches on the zoo bundle's `kind`:
- semantic — CE(+ignore_index 255) + 0.5·Dice loss, mIoU-monitored
  checkpoints, confusion matrix + per-class IoU bars;
- instance — same loop shape as detection over
  `CocoInstanceSegmentation`, evaluated with COCO box mAP.

## `src/visionsuite/metrics/metrics.py`

- `DetectionEvaluator` — torchmetrics `MeanAveragePrecision`
  (pycocotools-convention) with per-class AP extraction mapped back to
  class names.
- `ClassificationEvaluator` — accumulates logits; computes accuracy, top-5,
  balanced accuracy, macro/micro F1, full confusion matrix, per-class
  precision/recall/F1/support, and raw PR-curve arrays per class.
- `SegmentationEvaluator` — streaming pixel confusion matrix → mIoU, mDice,
  pixel accuracy, per-class IoU. O(C²) memory, O(pixels) time.

## `src/visionsuite/logutils/logger.py`

`RunLogger` implements the **run-directory contract** everything else relies
on (config.json / train.log / tensorboard/ / metrics.jsonl / checkpoints/ /
plots/ / eval/). `scalars(step, split, **kv)` is the single logging call:
it writes TensorBoard scalars, appends a JSON line, and prints a formatted
console/file line. `read_metrics()`/`list_runs()` are the read side used by
plots, comparison and the web app. Anything that honours the contract —
including the Ultralytics harvester — is automatically first-class.

## `src/visionsuite/logutils/plots.py`

All matplotlib output (Agg backend, saved PNGs): metric curves from
metrics.jsonl, row-normalized confusion matrices with value annotations,
multi-class PR curves, sorted per-class bar charts with value labels
(AP/IoU/F1/instance counts), and the two comparison plots (bar chart across
runs; overlaid curves across runs).

## `src/visionsuite/inference/predictor.py`

`Predictor.from_run(run_dir)` rebuilds the exact model from the checkpoint's
embedded config + class names — no external bookkeeping. Then:

- detection/instance: automatic **tiling for images > 2048 px** with
  stitched global NMS; results carry class, confidence, xyxy box,
  **pixel area, % of image area** and centroid, plus per-class counts.
- classification: top-k with probabilities at the model's native size.
- semantic: full-resolution argmax mask + per-class pixel/percentage areas.
- `annotate()` draws colour-coded boxes/labels, mask overlays, or the
  predicted class.
- `predict_batch()` writes `results.json`, flat `results.csv`, annotated
  images, and `summary.json` (per class: count, mean confidence, total
  area) — the "standard reporting" bundle.

## `src/visionsuite/compare.py`

Scans completed runs, builds a leaderboard (CSV + Markdown) sorted by the
task's primary metric (mAP50 / balanced accuracy / mIoU), and renders
comparison bars for every shared metric plus overlaid train-loss and
val-metric curves. Output under `runs/_comparisons/<task>/`.

## `scripts/`

Thin CLIs over the library (each supports `--config <yaml>` with CLI
overrides, and `--list` where relevant):

- `convert_data.py` — any format → canonical (+ optional tiling).
- `prepare_pcb_dataset.py` — the whole PCB pipeline in one command.
- `train_detection.py` — routes Ultralytics models to their trainer
  (after `coco_to_yolo` export) and harvests `results.csv` into
  metrics.jsonl + `best_metrics.json`; everything else goes to the engine.
- `train_classification.py`, `train_segmentation.py` — engine frontends.
- `infer.py` — single image or batch with all reports.
- `compare_runs.py` — leaderboard + comparison plots.

## `webapp/app.py`

Streamlit, four pages (Dashboard / Compare / Single inference / Batch
inference), reading only the run-directory contract and the `Predictor`.
Models are cached with `st.cache_resource` so switching images is instant.
Batch results are downloadable as CSV/JSON/zip.

## `configs/`, `tests/`

Ready-to-edit YAML configs for the PCB dataset, and dependency-light pytest
smoke tests for converters, tiling geometry/box remapping and the
evaluators.
