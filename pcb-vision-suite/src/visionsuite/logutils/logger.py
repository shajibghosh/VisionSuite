"""Unified experiment logging.

Every training run gets a self-describing run directory::

    runs/<task>/<model>_<timestamp>/
        config.json          # full resolved config (reproducibility)
        train.log            # console mirror
        tensorboard/         # scalars, images, histograms, PR curves
        metrics.jsonl        # one JSON line per epoch (machine-readable)
        checkpoints/         # best.pt + last.pt
        plots/               # loss curves, confusion matrix, PR, per-class AP
        eval/                # final evaluation artifacts

``metrics.jsonl`` is the contract consumed by the comparison module and the
Streamlit web app — anything that appends valid lines to it (including the
Ultralytics harvester) shows up in the dashboard automatically.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


class RunLogger:
    def __init__(self, task: str, model_name: str, root: str | Path = "runs",
                 run_name: str | None = None):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / task / (run_name or f"{model_name}_{stamp}")
        for sub in ("tensorboard", "checkpoints", "plots", "eval"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.task, self.model_name = task, model_name
        self._metrics_file = (self.run_dir / "metrics.jsonl").open("a")
        self._tb = None
        self.log = logging.getLogger(str(self.run_dir))
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                "%H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout),
                  logging.FileHandler(self.run_dir / "train.log")):
            h.setFormatter(fmt)
            self.log.addHandler(h)

    # ------------------------------------------------------------------ #
    @property
    def tb(self):
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(self.run_dir / "tensorboard")
        return self._tb

    def save_config(self, cfg: dict) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2,
                                                             default=str))
        self.log.info("config: %s", json.dumps(cfg, default=str))

    def scalars(self, step: int, split: str, **kv) -> None:
        """Log to console + TensorBoard + metrics.jsonl in one call."""
        clean = {k: float(v) for k, v in kv.items() if v is not None}
        for k, v in clean.items():
            self.tb.add_scalar(f"{split}/{k}", v, step)
        line = {"step": step, "split": split, **clean, "time": time.time()}
        self._metrics_file.write(json.dumps(line) + "\n")
        self._metrics_file.flush()
        pretty = "  ".join(f"{k}={v:.4f}" for k, v in clean.items())
        self.log.info("[%s %d] %s", split, step, pretty)

    def image(self, tag: str, img, step: int = 0) -> None:
        self.tb.add_image(tag, img, step, dataformats="HWC")

    def close(self) -> None:
        self._metrics_file.close()
        if self._tb:
            self._tb.close()


def read_metrics(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def list_runs(root: str | Path = "runs", task: str | None = None) -> list[Path]:
    root = Path(root)
    tasks = [task] if task else ["detection", "classification", "segmentation"]
    out = []
    for t in tasks:
        d = root / t
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.is_dir()))
    return out
