"""VisionSuite — a unified fine-tuning, evaluation, and inference suite for
object detection, image classification, and segmentation.

Designed around three principles:

1. **Model zoo registries** — every task exposes a registry of 10+ SOTA
   architectures behind one factory function, so adding a model is a
   one-line registration, not a new training script.
2. **Universal data ingestion** — converters normalise any common annotation
   format (COCO / Pascal VOC / YOLO / CSV / LabelMe / ImageFolder / masks)
   into a single canonical format per task, so every model trains from the
   same dataloaders.
3. **First-class reporting** — console + file + TensorBoard logging,
   automatic plots (loss curves, PR curves, confusion matrices, per-class
   AP), run comparison reports, and a Streamlit web app for interactive
   single/batch inference.

Primary use case: the WACV-2019 PCB Component Detection dataset
(https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection).
"""

__version__ = "1.0.0"

TASKS = ("detection", "classification", "segmentation")
