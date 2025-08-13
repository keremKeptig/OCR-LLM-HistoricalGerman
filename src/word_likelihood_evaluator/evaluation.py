from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Tuple
import os
import math
import statistics as stats
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate


class ErrorScorerWithGt(ABC):
    @abstractmethod
    def calculate_page_error_score(
        self,
        gt_image_path: Path,
        gt_xml_path: Path,
        pred_image_path: Path,
        pred_xml_path: Path,
    ) -> float:
        raise NotImplementedError


class ErrorScorerFromPred(ABC):
    @abstractmethod
    def calculate_page_error_score(self, image_path: Path, xml_path: Path) -> float:
        raise NotImplementedError


def calculate_error_for_dataset(
    path_to_dataset: str | Path,
    gt_scorer: ErrorScorerWithGt,
    pred_scorer: ErrorScorerFromPred,
) -> Tuple[List[float], List[float]]:
    dataset_root = Path(path_to_dataset)
    gt_folder = dataset_root / "d2_0001-0100_with_marginalia"
    pred_folder = dataset_root / "d2_0001-0100_without_marginalia"
    gt_scorer_errors: List[float] = []
    pred_scorer_errors: List[float] = []

    for gt_xml in gt_folder.glob("*.xml"):
        base_name = gt_xml.stem
        gt_image = gt_folder / f"{base_name}.bin.png"
        pred_image = pred_folder / f"{base_name}.bin.png"
        pred_xml = pred_folder / f"{base_name}.xml"
        if not (gt_image.exists() and pred_image.exists() and pred_xml.exists()):
            continue
        gt_error = gt_scorer.calculate_page_error_score(
            gt_image, gt_xml, pred_image, pred_xml
        )
        pred_error = pred_scorer.calculate_page_error_score(pred_image, pred_xml)
        gt_scorer_errors.append(gt_error)
        pred_scorer_errors.append(pred_error)
    return gt_scorer_errors, pred_scorer_errors


def _describe(errors: Sequence[float]) -> dict[str, float]:
    if not errors:
        return {k: math.nan for k in ("count", "mean", "median", "stdev", "min", "max", "rmse")}
    return {
        "count": len(errors),
        "mean": stats.mean(errors),
        "median": stats.median(errors),
        "stdev": stats.stdev(errors) if len(errors) > 1 else 0.0,
        "min": min(errors),
        "max": max(errors),
    }


def statistics(
    gt_scorer_errors: Sequence[float],
    pred_scorer_errors: Sequence[float],
    show_plots: bool = True,
) -> None:
    if not gt_scorer_errors or not pred_scorer_errors:
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
            differences = [(a - b) for a, b in zip(gt_scorer_errors, pred_scorer_errors)]
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
        plt.savefig("error_distribution_histogram.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

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
                linestyle="--", color="gray", label="Ideal match"
            )
            plt.legend()
            plt.savefig("gt_vs_pred_scatter.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()







