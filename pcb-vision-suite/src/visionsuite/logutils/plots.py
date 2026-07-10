"""Standard performance plots, saved as PNG into ``<run>/plots/`` and (when a
RunLogger is supplied) mirrored into TensorBoard.

Included:
- training/validation loss & metric curves (from metrics.jsonl)
- confusion matrix (classification / segmentation pixel-level)
- precision-recall curves (one per class + micro average)
- per-class AP / IoU / F1 bar charts (crucial for the skewed PCB classes)
- class distribution histogram of a dataset
- multi-run comparison bars and curve overlays
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------ curves ------------------------------------ #
def training_curves(metrics_rows: list[dict], out_dir: str | Path) -> list[Path]:
    """One figure per scalar key found in metrics.jsonl, train vs val."""
    out_dir = Path(out_dir)
    keys = set()
    for r in metrics_rows:
        keys.update(k for k in r if k not in ("step", "split", "time"))
    paths = []
    for k in sorted(keys):
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for split in ("train", "val", "test"):
            xs = [r["step"] for r in metrics_rows if r["split"] == split and k in r]
            ys = [r[k] for r in metrics_rows if r["split"] == split and k in r]
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, label=split)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("epoch"); ax.set_ylabel(k); ax.set_title(k); ax.legend()
        paths.append(_save(fig, out_dir / f"curve_{k}.png"))
    return paths


# -------------------------- confusion matrix ------------------------------ #
def confusion_matrix_plot(cm: np.ndarray, class_names: list[str],
                          out: str | Path, normalize: bool = True) -> Path:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35),) * 2)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    if n <= 25:
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, fraction=0.046)
    return _save(fig, Path(out))


# ------------------------------ PR curves --------------------------------- #
def pr_curves(precisions: dict[str, np.ndarray], recalls: dict[str, np.ndarray],
              out: str | Path, max_classes: int = 12) -> Path:
    """precisions/recalls: class name -> arrays of matching length."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, name in enumerate(list(precisions)[:max_classes]):
        ax.plot(recalls[name], precisions[name], lw=1.2, label=name)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Precision–Recall")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, Path(out))


# ----------------------------- bar charts --------------------------------- #
def per_class_bars(values: dict[str, float], out: str | Path,
                   metric_name: str = "AP") -> Path:
    names = list(values); vals = [values[n] for n in names]
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]; vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.32), 4))
    bars = ax.bar(range(len(names)), vals, color="#3b7dd8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Per-class {metric_name}")
    ax.bar_label(bars, fmt="%.2f", fontsize=6, rotation=90, padding=2)
    return _save(fig, Path(out))


def class_distribution(counts: dict[str, int], out: str | Path) -> Path:
    p = per_class_bars({k: float(v) for k, v in counts.items()}, out, "instances")
    return p


# --------------------------- run comparison ------------------------------- #
def comparison_bar(run_scores: dict[str, float], metric: str,
                   out: str | Path) -> Path:
    names = list(run_scores); vals = [run_scores[n] for n in names]
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]; vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7), 4))
    bars = ax.bar(names, vals, color=plt.cm.viridis(np.linspace(0.2, 0.85, len(names))))
    ax.set_ylabel(metric); ax.set_title(f"Model comparison — {metric}")
    ax.bar_label(bars, fmt="%.3f", fontsize=7)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    return _save(fig, Path(out))


def comparison_curves(run_metrics: dict[str, list[dict]], key: str,
                      split: str, out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, rows in run_metrics.items():
        xs = [r["step"] for r in rows if r["split"] == split and key in r]
        ys = [r[key] for r in rows if r["split"] == split and key in r]
        if xs:
            ax.plot(xs, ys, label=run, lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel(key)
    ax.set_title(f"{split}/{key} across runs")
    ax.legend(fontsize=7)
    return _save(fig, Path(out))
