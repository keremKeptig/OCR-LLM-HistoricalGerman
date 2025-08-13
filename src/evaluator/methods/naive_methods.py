import random
from pathlib import Path

from src.evaluator.page_parser import *
from src.evaluator.evaluation import ErrorScorerWithGt, ErrorScorerFromPred

class NaivePredScorer(ErrorScorerFromPred):
    def calculate_detailed_error_scores(self, pred_page: Page) -> Page:
        for word in pred_page.iter_words():
            score = random.uniform(0.0, 1.0)
            word.word_error_score = score
            word.error_type = ErrorType.ERROR if score > 0.5 else ErrorType.NONE
        return pred_page