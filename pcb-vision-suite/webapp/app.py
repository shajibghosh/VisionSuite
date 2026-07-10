#!/usr/bin/env python
"""VisionSuite web app (Streamlit).

Four pages:

1. **Dashboard** — every training run: config, live metric curves
   (read from metrics.jsonl), saved plots (confusion matrices, PR curves,
   per-class AP/IoU/F1), and best-checkpoint metrics.
2. **Compare models** — leaderboard table + comparison bar charts and
   overlaid validation curves across all runs of a task.
3. **Single inference** — upload one image (or pick a sample), run any
   trained checkpoint, view annotated output and a per-object table with
   class, confidence, bbox, pixel area and % area.
4. **Batch inference** — upload many images or point at a server folder;
   get a gallery of annotated results, per-class summary (counts, mean
   confidence, total area) and downloadable results.csv / results.json.

Run:  streamlit run webapp/app.py -- --runs-root runs
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionsuite.compare import PRIMARY, collect_runs, compare  # noqa: E402
from visionsuite.logutils.logger import list_runs, read_metrics  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    known, _ = ap.parse_known_args()
    return known


ARGS = _args()
st.set_page_config(page_title="VisionSuite", layout="wide", page_icon="🔬")


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(run_dir: str):
    from visionsuite.inference.predictor import Predictor
    return Predictor.from_run(run_dir)


def _runs(task=None):
    return [str(p) for p in list_runs(ARGS.runs_root, task)]


def metrics_df(run_dir):
    rows = read_metrics(run_dir)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =========================== pages =========================== #
def page_dashboard():
    st.header("📊 Training runs dashboard")
    runs = _runs()
    if not runs:
        st.info("No runs found. Train something first, e.g.\n\n"
                "`python scripts/train_detection.py --model faster_rcnn_v2 "
                "--ann data/pcb/detection/annotations_tiled.json`")
        return
    run = st.selectbox("Run", runs, format_func=lambda p: f"{Path(p).parent.name} / {Path(p).name}")
    run = Path(run)

    cfg_file = run / "config.json"
    best_file = run / "eval" / "best_metrics.json"
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Config")
        if cfg_file.exists():
            st.json(json.loads(cfg_file.read_text()))
    with c2:
        st.subheader("Best checkpoint metrics")
        if best_file.exists():
            best = json.loads(best_file.read_text())
            scalars = {k: v for k, v in best.items() if isinstance(v, (int, float))}
            cols = st.columns(min(6, max(1, len(scalars))))
            for i, (k, v) in enumerate(scalars.items()):
                cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, float) else v)
            for k, v in best.items():
                if isinstance(v, dict):
                    with st.expander(f"per-class: {k}"):
                        st.dataframe(pd.DataFrame(v, index=[k]).T
                                     if not isinstance(next(iter(v.values()), None), dict)
                                     else pd.DataFrame(v).T)

        df = metrics_df(run)
        if not df.empty:
            st.subheader("Metric curves")
            keys = [c for c in df.columns if c not in ("step", "split", "time")]
            key = st.selectbox("metric", keys)
            piv = df.dropna(subset=[key]).pivot_table(
                index="step", columns="split", values=key)
            st.line_chart(piv)

    plots = sorted((run / "plots").glob("*.png"))
    if plots:
        st.subheader("Saved plots")
        cols = st.columns(3)
        for i, p in enumerate(plots):
            cols[i % 3].image(str(p), caption=p.stem, use_container_width=True)

    st.caption(f"TensorBoard: `tensorboard --logdir {run / 'tensorboard'}`  ·  "
               f"log file: `{run / 'train.log'}`")


def page_compare():
    st.header("⚖️ Model comparison")
    task = st.radio("Task", ["detection", "classification", "segmentation"],
                    horizontal=True)
    rows = collect_runs(ARGS.runs_root, task)
    if len(rows) == 0:
        st.info(f"No completed {task} runs yet.")
        return
    primary = PRIMARY[task]
    table = pd.DataFrame([{"run": r["run"], "model": r["model"], **r["best"]}
                          for r in rows])
    if primary in table.columns:
        table = table.sort_values(primary, ascending=False)
    st.dataframe(table, use_container_width=True)
    st.download_button("Download leaderboard CSV",
                       table.to_csv(index=False), f"leaderboard_{task}.csv")

    if len(rows) >= 2 and st.button("Generate comparison plots"):
        info = compare(task, ARGS.runs_root)
        for p in info.get("plots", []):
            if Path(p).exists():
                st.image(p, use_container_width=True)

    metric_cols = [c for c in table.columns if c not in ("run", "model", "epoch")]
    if metric_cols:
        m = st.selectbox("Chart metric", metric_cols,
                         index=metric_cols.index(primary) if primary in metric_cols else 0)
        st.bar_chart(table.set_index("run")[m])


def _objects_table(res):
    if res.get("objects"):
        df = pd.DataFrame(res["objects"])
        df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(df["bbox_xyxy"].tolist())
        return df.drop(columns=["bbox_xyxy", "centroid"])
    return pd.DataFrame()


def page_single():
    st.header("🔎 Single-image inference")
    runs = _runs()
    if not runs:
        st.info("Train a model first.")
        return
    run = st.selectbox("Trained run", runs,
                       format_func=lambda p: f"{Path(p).parent.name} / {Path(p).name}")
    thr = st.slider("Score threshold", 0.05, 0.95, 0.35, 0.05)
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if up is None:
        return
    img = Image.open(up).convert("RGB")
    if st.button("Run inference", type="primary"):
        pred = load_predictor(run)
        with st.spinner("Predicting… (large boards are tiled automatically)"):
            res = pred.predict_image(img, score_thr=thr)
        c1, c2 = st.columns(2)
        c1.image(img, caption="input", use_container_width=True)
        annotated = pred.annotate(img, res)
        c2.image(annotated, caption="prediction", use_container_width=True)

        st.caption(f"latency: {res['latency_s']}s · image {res['image_size']}")
        if res["task"] == "detection":
            st.subheader(f"{len(res['objects'])} objects")
            st.dataframe(_objects_table(res), use_container_width=True)
            st.bar_chart(pd.Series(res["counts"], name="count"))
        elif res["task"] == "classification":
            st.subheader(f"→ {res['prediction']} ({res['confidence']:.3f})")
            st.bar_chart(pd.DataFrame(res["topk"]).set_index("class")["prob"])
        else:
            st.subheader("Class areas")
            st.dataframe(pd.DataFrame(res["class_areas"]).T)
        res.pop("_mask", None)
        st.download_button("Download result JSON", json.dumps(res, indent=2),
                           "result.json")


def page_batch():
    st.header("🗂️ Batch inference")
    runs = _runs()
    if not runs:
        st.info("Train a model first.")
        return
    run = st.selectbox("Trained run", runs,
                       format_func=lambda p: f"{Path(p).parent.name} / {Path(p).name}")
    thr = st.slider("Score threshold", 0.05, 0.95, 0.35, 0.05, key="bthr")
    ups = st.file_uploader("Images", accept_multiple_files=True,
                           type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    folder = st.text_input("…or a server-side folder path", "")
    if not st.button("Run batch", type="primary"):
        return

    pred = load_predictor(run)
    out_dir = Path("webapp_batch_out")
    if ups:
        tmp = Path("webapp_uploads"); tmp.mkdir(exist_ok=True)
        paths = []
        for u in ups:
            p = tmp / u.name
            p.write_bytes(u.getvalue())
            paths.append(p)
        images = paths
    elif folder and Path(folder).is_dir():
        images = folder
    else:
        st.error("Upload images or give a valid folder.")
        return

    with st.spinner("Running batch inference…"):
        out = pred.predict_batch(images, out_dir, score_thr=thr)

    st.success(f"Processed {out['summary']['num_images']} images")
    if out["summary"]["per_class"]:
        st.subheader("Per-class summary")
        st.dataframe(pd.DataFrame(out["summary"]["per_class"]).T)

    ann_dir = out_dir / "annotated"
    imgs = sorted(ann_dir.glob("*"))[:24]
    if imgs:
        st.subheader("Annotated gallery")
        cols = st.columns(4)
        for i, p in enumerate(imgs):
            cols[i % 4].image(str(p), caption=p.name, use_container_width=True)

    st.download_button("results.csv", (out_dir / "results.csv").read_bytes(),
                       "results.csv")
    st.download_button("results.json", (out_dir / "results.json").read_bytes(),
                       "results.json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ann_dir.glob("*"):
            z.write(p, p.name)
    st.download_button("annotated images (zip)", buf.getvalue(), "annotated.zip")


PAGES = {"Dashboard": page_dashboard, "Compare models": page_compare,
         "Single inference": page_single, "Batch inference": page_batch}

st.sidebar.title("🔬 VisionSuite")
st.sidebar.caption("Detection · Classification · Segmentation\n\n"
                   "PCB component analysis toolkit")
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
