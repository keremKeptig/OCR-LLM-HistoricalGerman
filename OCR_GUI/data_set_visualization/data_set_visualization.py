import io
import logging
from math import sqrt
import os
from pathlib import Path
import sys
from threading import Lock      
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, Response, current_app, jsonify, render_template, request
from scipy.stats import pearsonr, spearmanr
# --------------------------------------------------------------------------- #
# 1.  DATA LOADING  (at import time: happens exactly once)
# --------------------------------------------------------------------------- #
# adjust this to wherever your data lives
DATASET_DIR = Path("data")
matplotlib.use("Agg")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from src.evaluator.page_parser import load_pages
from src.evaluator.page_parser import Page     

_PAGESETS: dict[str, List[Page]] = {
    "Supervised Learning": load_pages(DATASET_DIR / "supervised_learning_approach"),
    "Chunking Approach": load_pages(DATASET_DIR / "chunking_approach"),
    "Manually annotated": load_pages(DATASET_DIR / "manual_annotation_approach"),
    "Hybrid-GT-Score":    load_pages(DATASET_DIR / "hybrid_gt_scorer_approach"),
}

APPROACH_OPTIONS = list(_PAGESETS.keys())  # keep insertion order

# --------------------------------------------------------------------------- #
# 2.  FLASK BLUEPRINT
# --------------------------------------------------------------------------- #
data_viz_bp = Blueprint(
    "data_viz",
    __name__,
    template_folder="../templates",
)

# ---------- MAIN PAGE ------------------------------------------------------ #
@data_viz_bp.route("/data_set_visualization")
def data_set_visualization():
    # send one *initial* metric set so the table isn’t empty;
    # we’ll default to the first two approaches
    init_stats = _calculate_statistics_for_page_sets(
        _PAGESETS[APPROACH_OPTIONS[0]],
        _PAGESETS[APPROACH_OPTIONS[1]],
        draw_plots=False,
    )
    return render_template(
        "data_set_visualization.html",
        approach_options=APPROACH_OPTIONS,
        metrics=init_stats,
    )

# ---------- METRIC ENDPOINT ------------------------------------------------ #
@data_viz_bp.route("/metrics.json")
def metrics_json():
    a1 = request.args.get("a1")
    a2 = request.args.get("a2")
    if a1 not in _PAGESETS or a2 not in _PAGESETS:
        return jsonify({"error": "Unknown approach"}), 400

    full_stats = _calculate_statistics_for_page_sets(
        _PAGESETS[a1], _PAGESETS[a2], draw_plots=False
    )
    numeric_only = {k: v for k, (_, v) in full_stats.items()}
    return jsonify(numeric_only)


_MPL_LOCK = Lock()               #  NEW ─ global lock around matplotlib

# ────────────────────────────────────────────────────────────────────────────
#  PLOT ENDPOINT
# ────────────────────────────────────────────────────────────────────────────
@data_viz_bp.route("/plot/<int:plot_id>.png")
def plot_png(plot_id: int):
    a1 = request.args.get("a1", APPROACH_OPTIONS[0])
    a2 = request.args.get("a2", APPROACH_OPTIONS[1])
    current_app.logger.debug("plot %s requested: a1=%s  a2=%s", plot_id, a1, a2)

    if a1 not in _PAGESETS or a2 not in _PAGESETS:
        return _empty_png("unknown\nparams"), 400

    pages1 = _PAGESETS[a1]
    pages2 = _PAGESETS[a2]
    arr1, arr2, _ = _paired_arrays(pages1, pages2)

    try:
        with _MPL_LOCK:                          # ← THE CRITICAL SECTION
            buf = io.BytesIO()
            if plot_id == 1:
                fig = _draw_scatter(arr1, arr2, title=f"{a1}  vs  {a2}")
            else:
                fig = _draw_overlay(arr1, arr2, title=f"{a1}  vs  {a2}")

            fig.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)                       # close *only* this figure
    except Exception:                            # any matplotlib hiccup
        current_app.logger.exception("Could not create plot")
        return _empty_png("internal\nerror"), 500

    buf.seek(0)
    resp = Response(buf.getvalue(), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

# --------------------------------------------------------------------------- #
# 3.  CORE STATISTICS + PLOTTING HELPERS
# --------------------------------------------------------------------------- #
def _paired_arrays(
    page_set_1: Sequence[Page],
    page_set_2: Sequence[Page]
):
    scores1 = {p.name: p.calculate_total_relative_error_score() for p in page_set_1}
    scores2 = {p.name: p.calculate_total_relative_error_score() for p in page_set_2}
    common = list(set(scores1) & set(scores2))
    if not common:
        raise ValueError("No overlapping pages.")
    arr1 = np.array([scores1[n] for n in common], dtype=float)
    arr2 = np.array([scores2[n] for n in common], dtype=float)
    return arr1, arr2, len(common)

def _calculate_statistics_for_page_sets(p1, p2, *, draw_plots=False):
    arr1, arr2, n = _paired_arrays(p1, p2)
    return {
        "rmse":         ("Root Mean Square Error (RMSE)", float(np.sqrt(np.mean((arr1-arr2)**2)))),
        "spearman_r":   ("Spearman Correlation",          float(spearmanr(arr1, arr2)[0])),
        "pearson_r":    ("Pearson Correlation",           float(pearsonr(arr1,  arr2)[0])),
        "mean1":        ("Mean – Approach 1",             float(arr1.mean())),
        "mean2":        ("Mean – Approach 2",             float(arr2.mean())),
        "n_pages":      ("Amount of compared Pages",       n),
    }

# ----------------- Plot helpers ------------------------------------------- #
# ────────────────────────────────────────────────────────────────────────────
# ── plot helpers ───────────────────────────────────────────
def _draw_scatter(arr1, arr2, *, title="Paired error scores"):
    # ❶  wider aspect ❷ higher dpi keeps it crisp when enlarged
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.scatter(arr1, arr2, alpha=0.7, s=14)
    lo, hi = min(arr1.min(), arr2.min()), max(arr1.max(), arr2.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set_xlabel("Error score – set 1")
    ax.set_ylabel("Error score – set 2")
    ax.set_title(title)
    return fig


def _draw_overlay(arr1, arr2, *, title="Error overlay (set 1 sorted)"):
    sort_idx = np.argsort(arr1)
    x = np.arange(len(arr1))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)   # same change here
    ax.plot(x, arr1[sort_idx], marker="o", markersize=4, label="set 1 (sorted)")
    ax.plot(x, arr2[sort_idx], marker="o", markersize=4, label="set 2 (paired)")
    ax.set_xlabel("Page index")
    ax.set_ylabel("Error score")
    ax.set_title(title)
    ax.legend()
    return fig



def _empty_png(msg: str = "") -> Response:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, wrap=True)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")
