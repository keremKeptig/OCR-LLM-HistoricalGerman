from pathlib import Path
import json

from src.evaluator.evaluation import ErrorScorerWithGt


class ManualGTScorer(ErrorScorerWithGt):

    def calculate_page_error_score(
        self,
        json_path: Path,
        gt_image_path: Path,
        gt_xml_path: Path,
        pred_image_path: Path,
        pred_xml_path: Path,
    ) -> float:
        error_score = 0
        json_file_path = json_path

        if json_file_path.exists():
            with open(json_file_path, "r") as f:
                json_data = json.load(f)
                error_score = json_data.get("error_ratio")
        else:
            print("File not found:", json_file_path)

        return error_score
