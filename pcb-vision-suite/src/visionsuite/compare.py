"""Cross-run comparison and leaderboards.

Scans ``runs/<task>/*`` for the suite's run-directory contract
(``config.json``, ``metrics.jsonl``, ``eval/best_metrics.json``) and builds:

- a leaderboard table (CSV + markdown) sorted by the task's primary metric
- comparison bar charts for every shared metric
- overlaid validation curves across runs

Primary metrics: detection -> mAP50, classification -> balanced_accuracy,
segmentation -> mIoU (falls back to mAP50 for instance models).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .logutils.logger import list_runs, read_metrics
from .logutils.plots import comparison_bar, comparison_curves

PRIMARY = {"detection": "mAP50", "classification": "balanced_accuracy",
           "segmentation": "mIoU"}


def collect_runs(root: str | Path = "runs", task: str | None = None) -> list[dict]:
    rows = []
    for run in list_runs(root, task):
        best_file = run / "eval" / "best_metrics.json"
        cfg_file = run / "config.json"
        if not best_file.exists():
            continue
        try:
            best = json.loads(best_file.read_text())
            cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        except json.JSONDecodeError:
            continue
        rows.append({
            "run": run.name, "run_dir": str(run), "task": run.parent.name,
            "model": cfg.get("model", run.name.split("_")[0]),
            "config": cfg,
            "best": {k: v for k, v in best.items() if isinstance(v, (int, float))},
            "per_class": {k: v for k, v in best.items() if isinstance(v, dict)},
        })
    return rows


def compare(task: str, root: str | Path = "runs",
            out_dir: str | Path = "runs/_comparisons") -> dict:
    out_dir = Path(out_dir) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(root, task)
    if not rows:
        return {"task": task, "runs": 0}
    primary = PRIMARY[task]
    if not any(primary in r["best"] for r in rows):
        primary = "mAP50" if task == "segmentation" else primary
    rows.sort(key=lambda r: r["best"].get(primary, -1), reverse=True)

    # union of scalar metric keys
    keys = sorted({k for r in rows for k in r["best"] if k != "epoch"})
    with (out_dir / "leaderboard.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model"] + keys)
        for r in rows:
            w.writerow([r["run"], r["model"]] +
                       [round(r["best"].get(k, float("nan")), 4) for k in keys])

    md = ["| run | model | " + " | ".join(keys) + " |",
          "|" + "---|" * (len(keys) + 2)]
    for r in rows:
        md.append("| " + r["run"] + " | " + r["model"] + " | " +
                  " | ".join(f"{r['best'].get(k, float('nan')):.4f}"
                             for k in keys) + " |")
    (out_dir / "leaderboard.md").write_text("\n".join(md))

    plots = []
    for k in keys:
        scores = {r["run"]: r["best"][k] for r in rows if k in r["best"]}
        if len(scores) >= 2:
            plots.append(str(comparison_bar(scores, k, out_dir / f"cmp_{k}.png")))
    run_metrics = {r["run"]: read_metrics(r["run_dir"]) for r in rows}
    curve_keys = {"detection": ["mAP50", "loss"],
                  "classification": ["balanced_accuracy", "loss"],
                  "segmentation": ["mIoU", "loss"]}[task]
    for k in curve_keys:
        split = "train" if k == "loss" else "val"
        plots.append(str(comparison_curves(run_metrics, k, split,
                                           out_dir / f"curves_{split}_{k}.png")))
    return {"task": task, "runs": len(rows), "primary_metric": primary,
            "best_run": rows[0]["run"], "leaderboard": str(out_dir / "leaderboard.csv"),
            "plots": plots}
