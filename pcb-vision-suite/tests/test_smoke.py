"""Dependency-light smoke tests: converters, tiling geometry, metrics.

Run: python -m pytest tests/ -q   (torch/numpy/PIL required; no GPUs, no
model downloads — model-zoo construction is exercised only if the heavy
libraries are installed.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.data.converters import convert, detect_format, ir_to_coco
from visionsuite.data.converters.base import DatasetIR, ImageRecord, Instance
from visionsuite.data.tiling import _tile_grid, tile_coco_dataset
from visionsuite.metrics.metrics import (ClassificationEvaluator,
                                         SegmentationEvaluator)


def _fake_voc(tmp_path):
    img = tmp_path / "board1.jpg"
    Image.new("RGB", (200, 150), "green").save(img)
    (tmp_path / "board1.xml").write_text(f"""
<annotation><filename>board1.jpg</filename>
<size><width>200</width><height>150</height></size>
<object><name>resistor</name>
<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>50</ymax></bndbox>
</object></annotation>""")
    return tmp_path


def test_detect_and_convert_voc(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    src = _fake_voc(src)
    out = tmp_path / "out"
    ann = convert(src, out, "detection")
    data = json.loads(Path(ann).read_text())
    assert len(data["images"]) == 1
    assert data["annotations"][0]["bbox"] == [10.0, 20.0, 50.0, 30.0]
    assert data["categories"][0]["name"] == "resistor"


def test_tile_grid_covers_image():
    grid = _tile_grid(3000, 2000, 1024, 0.2)
    xs = {x for x, _ in grid}; ys = {y for _, y in grid}
    assert max(xs) + 1024 >= 3000 and max(ys) + 1024 >= 2000


def test_tiling_remaps_boxes(tmp_path):
    ir = DatasetIR()
    img = tmp_path / "big.jpg"
    Image.new("RGB", (2048, 1024), "black").save(img)
    ir.images.append(ImageRecord(str(img), 2048, 1024, instances=[
        Instance("cap", (1500.0, 100.0, 60.0, 40.0))]))
    ann = ir_to_coco(ir, tmp_path / "ann.json")
    tiled = tile_coco_dataset(tmp_path / "ann.json", tmp_path, tile=1024,
                              overlap=0.0)
    data = json.loads(Path(tiled).read_text())
    assert len(data["annotations"]) >= 1
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        assert 0 <= x and x + w <= 1024 + 1e-6


def test_classification_evaluator():
    ev = ClassificationEvaluator(["a", "b", "c"])
    logits = torch.tensor([[5., 0, 0], [0, 5., 0], [0, 5., 0], [0, 0, 5.]])
    ev.update(logits, torch.tensor([0, 1, 2, 2]))
    res = ev.compute()
    assert abs(res["accuracy"] - 0.75) < 1e-6
    assert res["confusion_matrix"].sum() == 4


def test_segmentation_evaluator():
    ev = SegmentationEvaluator(["bg", "fg"])
    logits = torch.zeros(1, 2, 4, 4); logits[0, 1] = 1  # predict all fg
    target = torch.ones(1, 4, 4, dtype=torch.long)
    ev.update(logits, target)
    assert ev.compute()["mIoU"] == 1.0
