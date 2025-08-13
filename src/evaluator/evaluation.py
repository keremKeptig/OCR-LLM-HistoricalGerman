from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Tuple
import os
import math
import statistics as stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from src.evaluator.page_parser import *
from dataclasses import dataclass
from math import sqrt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

class ErrorScorerWithGt(ABC):
    @abstractmethod
    def calculate_detailed_error_scores(
            self,
            gt_page: Page,
            pred_page: Page,
    ) -> Page:  # return the (error) annotated pred_page
        raise NotImplementedError


class ErrorScorerFromPred(ABC):

    @abstractmethod
    def calculate_detailed_error_scores(
            self,
            pred_page: Page,
    ) -> Page:  # return the (error) annotated pred_page
        raise NotImplementedError


def calculate_annotated_pages_with_gt_scorer(path_to_dataset: Path,
        gt_scorer: ErrorScorerWithGt,) -> (Page, Page):
    dataset_root = Path(path_to_dataset)
    gt_folder = dataset_root / "d2_0001-0100_with_marginalia"
    pred_folder = dataset_root / "d2_0001-0100_without_marginalia"

    original_gt_pages: List[Page] = []
    annotated_by_gt: List[Page] = []

    for gt_xml in gt_folder.glob("*.xml"):
        base = gt_xml.stem
        pred_xml = pred_folder / f"{base}.xml"

        if not pred_xml.exists():
            continue

        gt_page = parse_raw_page_from_xml(gt_xml)
        pred_page_for_gt = parse_raw_page_from_xml(pred_xml)

        annotated_gt = gt_scorer.calculate_detailed_error_scores(gt_page, pred_page_for_gt)

        annotated_by_gt.append(annotated_gt)
        original_gt_pages.append(gt_page)

    return annotated_by_gt, original_gt_pages

def calculate_annotated_pages_with_pred_scorer(path_to_dataset: Path,
    pred_scorer: ErrorScorerWithGt,) -> Page:
    dataset_root = Path(path_to_dataset)
    pred_folder = dataset_root / "d2_0001-0100_without_marginalia"

    annotated_by_pred: List[Page] = []

    for pred_xml in pred_folder.glob("*.xml"):

        if not pred_xml.exists():
            continue
        pred_page = parse_raw_page_from_xml(pred_xml)
        annotated_pred = pred_scorer.calculate_detailed_error_scores(pred_page)

        annotated_by_pred.append(annotated_pred)

    return annotated_by_pred



def calculate_detailed_annotated_pages(
        path_to_dataset: Path,
        gt_scorer: ErrorScorerWithGt,
        pred_scorer: ErrorScorerFromPred,
) -> Page:
    dataset_root = Path(path_to_dataset)
    gt_folder = dataset_root / "d2_0001-0100_with_marginalia"
    pred_folder = dataset_root / "d2_0001-0100_without_marginalia"

    original_gt_pages: List[Page] = []
    annotated_by_gt: List[Page] = []
    annotated_by_pred: List[Page] = []

    for gt_xml in gt_folder.glob("*.xml"):
        base = gt_xml.stem
        pred_xml = pred_folder / f"{base}.xml"

        if not pred_xml.exists():
            continue

        gt_page = parse_raw_page_from_xml(gt_xml)
        pred_page_for_gt = parse_raw_page_from_xml(pred_xml)
        pred_page_for_pred = parse_raw_page_from_xml(pred_xml)

        annotated_gt = gt_scorer.calculate_detailed_error_scores(gt_page, pred_page_for_gt)
        annotated_pred = pred_scorer.calculate_detailed_error_scores(pred_page_for_pred)

        annotated_by_gt.append(annotated_gt)
        annotated_by_pred.append(annotated_pred)
        original_gt_pages.append(gt_page)

    return annotated_by_gt, annotated_by_pred, original_gt_pages

from pathlib import Path
from typing import List, Optional

def print_error_visualization_for_pages(pages: List[Page], output_dir: Optional[Path | str]):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for page in pages:
        stem = (
            page.image_path.stem
            if page.image_path is not None
            else page.name
        )
        err_img = visualize_error_regions(page)
        pred_err_path = out_path / f"{stem}_error_overlay.png"
        print(f"Saving {pred_err_path}")
        err_img.save(pred_err_path)

def print_error_gradient_text_for_pages(pages: List[Page], output_dir: Optional[Path | str]):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for page in pages:
        stem = (
            page.image_path.stem
            if page.image_path is not None
            else page.name
        )
        err_img = visualize_text_errors(page)
        pred_err_path = out_path / f"{stem}_gradient_text.png"
        print(f"Saving {pred_err_path}")
        err_img.save(pred_err_path)




def print_region_layout_visualization_for_pages(pages: List[Page], output_dir: Optional[Path | str]):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for page in pages:
        stem = (
            page.image_path.stem
            if page.image_path is not None
            else page.name
        )
        layout_img = visualize_text_regions(page)
        layout_path = out_path / f"{stem}_region_layout_overlay.png"
        print(f"Saving {layout_path}")
        layout_img.save(layout_path)

def calulate_statistics_for_page_sets(
    page_set_1: List[Page],
    page_set_2: List[Page],
    draw_plots: bool = True):
    scores1 = _scores_by_name(page_set_1)
    scores2 = _scores_by_name(page_set_2)

    common_names = set(scores1) & set(scores2)
    if not common_names:
        raise ValueError("No pages with matching 'name' in both lists.")

    # Paired arrays for metrics
    arr1 = np.array([scores1[name] for name in common_names], dtype=float)
    arr2 = np.array([scores2[name] for name in common_names], dtype=float)

    # --- Scalar statistics ---------------------------------------------------
    rmse = float(sqrt(np.mean((arr1 - arr2) ** 2)))

    spear_r, spear_p = spearmanr(arr1, arr2)
    pear_r, pear_p = pearsonr(arr1, arr2)

    mean_1 = float(np.mean(list(scores1.values())))
    mean_2 = float(np.mean(list(scores2.values())))

    stats: Dict[str, float] = {
        "rmse": rmse,
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "mean_1": mean_1,
        "mean_2": mean_2,
        "n_pages": len(common_names),
    }

    print( "\n".join(f"{k:<10}: {v:.4g}" if isinstance(v, float) else f"{k:<10}: {v}" for k, v in stats.items() ) )


    # --- Top differences -----------------------------------------------------
    diffs = [
        (name, abs(scores1[name] - scores2[name]), scores1[name], scores2[name])
        for name in common_names
    ]
    top_diffs = sorted(diffs, key=lambda x: x[1], reverse=True)[:5]

    print("\nTop 5 pages by absolute error score difference:")
    for name, diff, s1, s2 in top_diffs:
        print(f"{name:<30} | Score1: {s1:.4f} | Score2: {s2:.4f} | Diff: {diff:.4f}")


    # --- Visualisations ------------------------------------------------------
    if draw_plots:
        _draw_scatter(arr1, arr2)
        _draw_overlay(arr1, arr2)

    return stats


def _scores_by_name(pages: Sequence[Page]) -> dict[str, float]:
    return {p.name: p.calculate_total_relative_error_score() for p in pages}



def _draw_scatter(arr1: np.ndarray, arr2: np.ndarray) -> None:
    """Scatter plot with identity line."""

    plt.figure()
    plt.scatter(arr1, arr2, alpha=0.7)
    plt.xlabel("Error score – set 1")
    plt.ylabel("Error score – set 2")
    plt.title("Paired error scores")

    lo = min(arr1.min(), arr2.min())
    hi = max(arr1.max(), arr2.max())
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)

    plt.tight_layout()
    plt.show()


def _draw_overlay(arr1: np.ndarray, arr2: np.ndarray) -> None:
    sort_idx = np.argsort(arr1)
    arr1_sorted = arr1[sort_idx]
    arr2_sorted = arr2[sort_idx]

    x = np.arange(len(arr1_sorted))

    plt.figure()
    plt.plot(x, arr1_sorted, marker="o", label="set 1 (sorted)")
    plt.plot(x, arr2_sorted, marker="o", label="set 2 (paired)")

    plt.xlabel("Page index (sorted by set 1 score)")
    plt.ylabel("Error score")
    plt.title("Per‑page error scores – overlay (set 1 sorted)")
    plt.legend()

    plt.tight_layout()
    plt.show()



# depricated
def depricated_statistics(
    gt_annotated_pages: List[Page],
    pred_annotated_pages: List[Page],
    original_gt_pages: List[Page],
    output_dir: Optional[Path | str] = None,
) -> None:
    # collect one scalar error score per page
    gt_scores = [p.calculate_total_relative_error_score() for p in gt_annotated_pages]
    pred_scores = [p.calculate_total_relative_error_score() for p in pred_annotated_pages]

    # averages
    avg_gt = sum(gt_scores) / len(gt_scores) if gt_scores else float("nan")
    avg_pred = sum(pred_scores) / len(pred_scores) if pred_scores else float("nan")

    # RMSE between GT and prediction
    import math
    rmse = math.sqrt(
        sum((g - p) ** 2 for g, p in zip(gt_scores, pred_scores)) / len(gt_scores)
    ) if gt_scores else float("nan")

    # console report -------------------------------------------------------
    print(f"\n––– page-level error statistics –––")
    print(f"GT pages   : μ = {avg_gt:.4f}")
    print(f"Pred pages : μ = {avg_pred:.4f}")
    print(f"RMSE       :   {rmse:.4f}\n")

    for gt_page_raw, gt_page_err, pred_page_err in zip(
        original_gt_pages, gt_annotated_pages, pred_annotated_pages
    ):

        if output_dir is not None: # visualized if output is set
            stem = (
                gt_page_raw.image_path.stem
                if gt_page_raw.image_path is not None
                else gt_page_raw.name
            )

            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # --- region overlays ------------------------------------------------
            gt_regions_img   = visualize_text_regions(gt_page_raw)
            pred_regions_img = visualize_text_regions(pred_page_err)

            gt_regions_path   = out_path / f"{stem}_1_text_regions_gt.png"
            pred_regions_path = out_path / f"{stem}_2_text_regions_pred.png"

            print(f"Saving {gt_regions_path}")
            gt_regions_img.save(gt_regions_path)

            print(f"Saving {pred_regions_path}")
            pred_regions_img.save(pred_regions_path)

            # --- error overlays -------------------------------------------------
            gt_err_img   = visualize_error_regions(gt_page_err)
            pred_err_img = visualize_error_regions(pred_page_err)

            gt_err_path   = out_path / f"{stem}_3_error_overlay_gt.png"
            pred_err_path = out_path / f"{stem}_4_error_overlay_pred.png"

            print(f"Saving {gt_err_path}")
            gt_err_img.save(gt_err_path)

            print(f"Saving {pred_err_path}")
            pred_err_img.save(pred_err_path)

            # --- gradient text-error visualisations -----------------------------
            gt_text_err_img = visualize_text_errors(gt_page_err)
            pred_text_err_img = visualize_text_errors(pred_page_err)

            gt_text_err_path = out_path / f"{stem}_5_text_errors_gt.png"
            pred_text_err_path = out_path / f"{stem}_6_text_errors_pred.png"

            print(f"Saving {gt_text_err_path}")
            gt_text_err_img.save(gt_text_err_path)

            print(f"Saving {pred_text_err_path}")
            pred_text_err_img.save(pred_text_err_path)


# deprecated
def calculate_error_for_json_GT(
        path_to_dataset: str | Path,
        gt_scorer: ErrorScorerWithGt,
        pred_scorer: ErrorScorerFromPred,
) -> Tuple[List[float], List[float]]:
    dataset_root = Path(path_to_dataset)
    gt_folder = dataset_root / "error_score_gt"
    pred_folder = dataset_root / "d2_0001-0100_without_marginalia"

    gt_scorer_errors: List[float] = []
    pred_scorer_errors: List[float] = []

    for gt_json in gt_folder.glob("*.json"):
        base_name = gt_json.stem

        json_path = gt_folder / f"{base_name}.json"
        pred_xml = pred_folder / f"{base_name}.xml"
        if not (pred_xml.exists()):
            continue
        gt_error = gt_scorer.calculate_page_error_score(
            json_path=json_path,
            gt_image_path=None,
            gt_xml_path=None,
            pred_image_path=None,
            pred_xml_path=pred_xml,
        )
        pred_error = pred_scorer.calculate_page_error_score(
            image_path=None, xml_path=pred_xml
        )
        gt_scorer_errors.append(gt_error)
        pred_scorer_errors.append(pred_error)
    return gt_scorer_errors, pred_scorer_errors


def _describe(errors: Sequence[float]) -> dict[str, float]:
    if not errors:
        return {
            k: math.nan
            for k in ("count", "mean", "median", "stdev", "min", "max", "rmse")
        }
    return {
        "count": len(errors),
        "mean": stats.mean(errors),
        "median": stats.median(errors),
        "stdev": stats.stdev(errors) if len(errors) > 1 else 0.0,
        "min": min(errors),
        "max": max(errors),
    }


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def statistics(
        gt_scorer_errors: Sequence[float],
        pred_scorer_errors: Sequence[float],
        show_plots: bool = True,
) -> None:
    if not gt_scorer_errors or not pred_scorer_errors:
        print("No errors to show statistics for.")
        return
    gt_stats = _describe(gt_scorer_errors)
    pred_stats = _describe(pred_scorer_errors)

    table = [
        ["Metric", "GT Scorer", "Pred Scorer"],
        *[
            [key.capitalize(), f"{gt_stats[key]:.4f}", f"{pred_stats[key]:.4f}"]
            for key in gt_stats
        ],
        [
            "MAE",
            f"{stats.mean(abs(e) for e in gt_scorer_errors):.4f}",
            f"{stats.mean(abs(e) for e in pred_scorer_errors):.4f}",
        ],
    ]

    if len(gt_scorer_errors) == len(pred_scorer_errors):
        try:
            corr = stats.correlation(gt_scorer_errors, pred_scorer_errors)
            table.append(["Correlation", f"{corr:.4f}", "—"])
            differences = [
                (a - b) for a, b in zip(gt_scorer_errors, pred_scorer_errors)
            ]
            pairwise_rmse = math.sqrt(stats.mean(d ** 2 for d in differences))
            table.append(["GT vs Pred RMSE", f"{pairwise_rmse:.4f}", "—"])
        except Exception:
            pass

    print(tabulate(table, headers="firstrow", tablefmt="grid"))

    if show_plots:
        # Plot 1: Histograms of errors
        plt.figure()
        plt.hist(gt_scorer_errors, bins=10, alpha=0.6, label="GT Scorer")
        plt.hist(pred_scorer_errors, bins=10, alpha=0.6, label="Pred Scorer")
        plt.title("Error Distributions")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Scatter plot of GT vs Pred error
        if len(gt_scorer_errors) == len(pred_scorer_errors):
            plt.figure()
            plt.scatter(gt_scorer_errors, pred_scorer_errors, alpha=0.7)
            plt.title("GT vs Pred Error (per page)")
            plt.xlabel("GT Scorer Error")
            plt.ylabel("Pred Scorer Error")
            plt.grid(True)
            plt.plot(
                [min(gt_scorer_errors), max(gt_scorer_errors)],
                [min(gt_scorer_errors), max(gt_scorer_errors)],
                linestyle="--",
                color="gray",
                label="Ideal match",
            )
            plt.legend()
            plt.show()

            window_size = 15
            smoothed_truth = moving_average(gt_scorer_errors, window_size)
            smoothed_pred = moving_average(pred_scorer_errors, window_size)

            x_values = range(window_size - 1, len(gt_scorer_errors))

            plt.figure(figsize=(12, 6))
            plt.plot(
                x_values,
                smoothed_truth,
                label="Ground Truth (smoothed)",
                linewidth=2,
                color="green",
            )
            plt.plot(
                x_values,
                smoothed_pred,
                label="Prediction (smoothed)",
                linewidth=2,
                color="red",
            )
            plt.title("Smoothed Ground Truth vs Prediction")
            plt.xlabel("Page index")
            plt.ylabel("Error score")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()
