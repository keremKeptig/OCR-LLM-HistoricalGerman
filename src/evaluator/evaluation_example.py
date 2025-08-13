from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from sympy import false
import json
from src.evaluator.evaluation import *
from src.evaluator.methods.naive_methods import  NaivePredScorer
from src.evaluator.methods.manual_gt_scorer import ManualGTScorer
from src.evaluator.methods.overlay_gt_scorer import OverlayGtScorer
from src.evaluator.methods.hybrid_overlay_gt_scorer import HybridOverlayGtScorer
from src.evaluator.page_parser import *

# define dataset path here (put the path of  where your data is for me its just in root/data
dataset_path = Path(__file__).resolve().parents[2]/ "data"

# example how to calculate pages with a gt scorer
gt_annotated_pages, original_gt_page = calculate_annotated_pages_with_gt_scorer(dataset_path, OverlayGtScorer())
# example how to calculate pages with a pred scorer
pred_annotated_pages = calculate_annotated_pages_with_pred_scorer(dataset_path, NaivePredScorer())
# example how to load from files
manually_annotated_pages = load_pages(Path(__file__).resolve().parents[2]/"data/manual_annotation_approach")
# example how to load data_subset (annotated errors with naive scorer
subset_pages = load_pages(Path(__file__).resolve().parents[2]/"data/chunking_approach")

# how to reduce to subset
reduce_to_common_page_by_name(gt_annotated_pages, subset_pages)

# if you want to visualize the error overlay
# print_error_visualization_for_pages(gt_annotated_pages, Path(__file__).resolve().parents[0]/"output/overlay_visualization")
# print_error_gradient_text_for_pages(gt_annotated_pages, Path(__file__).resolve().parents[0]/"output/overlay_visualization")
# print_region_layout_visualization_for_pages(gt_annotated_pages,  Path(__file__).resolve().parents[0]/"output/overlay_visualization")

# how to save pages
save_pages(gt_annotated_pages, Path(__file__).resolve().parents[2]/"data/your_approach")

# Statistics
calulate_statistics_for_page_sets(gt_annotated_pages, manually_annotated_pages, True)