from pathlib import Path
from src.word_likelihood_evaluator.evaluation import calculate_error_for_dataset, statistics
from src.word_likelihood_evaluator.methods.overlay_gt_scorer import OverlayGtScorer
from src.word_likelihood_evaluator.methods.pred_error_scorer import PredictionErrorScorer

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2] / "data"
    
    gt_folder = base_dir / "d2_0001-0100_with_marginalia"
    pred_folder = base_dir / "d2_0001-0100_without_marginalia"

    gt_scorer   = OverlayGtScorer()
    pred_scorer = PredictionErrorScorer(
        csv_folder=pred_folder,
        xml_folder=gt_folder
    )
    print(pred_scorer)
    gt_errs, pred_errs = calculate_error_for_dataset(base_dir, gt_scorer, pred_scorer)
    statistics(gt_errs, pred_errs)
