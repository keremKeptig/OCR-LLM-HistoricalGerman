import json
import random
import csv
from pathlib import Path

from src.evaluator.evaluation import ErrorScorerWithGt, ErrorScorerFromPred
from src.evaluator.page_parser import *


class LikelihoodErrorCount(ErrorScorerWithGt):

    def __init__(self, data_base_dir: Path = None):
        self.data_base_dir = data_base_dir

    def calculate_detailed_error_scores(self, pred_page: Page) -> Page:
        """
        Assigns error scores to words in `pred_page` based on matches in `pred_csv_path`
        and writes matched errors to `output_csv_path` as CSV: word,line_id,region_id.
        """
        error_list = self.error_list(pred_page)
        matched_errors = []  # To store rows for output CSV

        for region_id, line_id, word in pred_page.iter_words_with_line_id():
            word.word_error_score = 0 

            for err_word, color, err_line_id in error_list:
                if word.text in err_word and line_id == err_line_id:
                    if color == "red":
                        word.word_error_score = 1
                        word.error_type = ErrorType.ERROR
                        # only red ones are saved to CSV
                        matched_errors.append({
                        "word": word.text,
                        "line_id": line_id,
                        "region_id": region_id
                        })
                    elif color == "orange":
                        word.word_error_score = 0.5
                        word.error_type = ErrorType.NONE
                    else:
                        word.word_error_score = 0
                        word.error_type = ErrorType.NONE

                    # print(f"MATCH FOUND: {word.text} (Line: {line_id}) → {color}")                    
                    break  

        return pred_page

    def error_list(
        self, pred_page: Page
    ) -> Page:

        csv_path = self.data_base_dir / f"{pred_page.name}_ocr_errors.csv"

        error_list_with_line_id = []
        try:
            with open(csv_path, "r", encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                
                for row in csv_reader:
                    if row.get('word'):
                      entry = (row.get('word'), row.get('color'), row.get('line_id'))  # Tuple
                      error_list_with_line_id.append(entry)
                
    
            return error_list_with_line_id
            
        except FileNotFoundError:
            print(f"CSV file not found: {csv_path}")
            return 0.0
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return 0.0

    