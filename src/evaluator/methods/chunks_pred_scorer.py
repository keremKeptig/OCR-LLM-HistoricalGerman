from pathlib import Path
import pandas as pd
import math

import Levenshtein
from src.evaluator.page_parser import *
from src.evaluator.evaluation import ErrorScorerFromPred


class ChunksPredScorer(ErrorScorerFromPred):
    """Class to calculate detailed error scores for words in a page based on their perplexity values.
    It normalizes the perplexity values using a sigmoid function and assigns error types based on the normalized scores.

    Args:
        data_base_dir (Path): The base directory where the CSV files with perplexity values are stored.
    """

    def __init__(self, data_base_dir: Path = None):
        self.data_base_dir = data_base_dir

    def normalize_text(self, text: str) -> str:
        """Normalizes the text by converting it to lowercase and replacing specific characters.
        Args:
            text (str): The text to normalize.
        Returns:
            str: The normalized text.
        """
        return (
            text.lower()
            .replace("ſ", "s")
            .replace("ʒ", "z")
            .replace("-", "")
            .replace("ƶ", "z")
        )

    def sigmoid_normalize(self, perplexity, midpoint=100, steepness=0.1):
        """Applies a sigmoid function to normalize the perplexity value.
        Args:
            perplexity (float): The perplexity value to normalize.
            midpoint (float): The midpoint of the sigmoid function.
            steepness (float): The steepness of the sigmoid function.
        Returns:
            float: The normalized score between 0 and 1.
        """
        return 1 / (1 + math.exp(-steepness * (perplexity - midpoint)))

    def calculate_detailed_error_scores(self, pred_page: Page) -> Page:
        """Calculates detailed error scores for each word in the predicted page based on perplexity values.
        Args:
            pred_page (Page): The predicted page containing words to score.
        Returns:
            Page: The updated page with error scores and types assigned to each word.
        """
        page_words = list(pred_page.iter_words())

        csv_path = self.data_base_dir / f"{pred_page.name}.csv"

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            for word in page_words:
                word.word_error_score = 0.0
                word.error_type = ErrorType.NONE
            return pred_page

        csv_rows = list(df.itertuples(index=False))
        page_words_idx = 0
        csv_idx = 0

        while csv_idx < len(csv_rows) and page_words_idx < len(page_words):
            row = csv_rows[csv_idx]
            csv_word = str(row.word)
            ppl = float(row.perplexity)
            norm_score = self.sigmoid_normalize(ppl)

            word = page_words[page_words_idx]
            next_word = (
                page_words[page_words_idx + 1]
                if page_words_idx + 1 < len(page_words)
                else None
            )

            if word.text.endswith("⸗") and next_word:

                combined = word.text.rstrip("⸗") + next_word.text.replace("⸗", "")
                combined = self.normalize_text(combined)
                lev_dist = Levenshtein.distance(combined, csv_word.lower())
                if lev_dist <= max(1, int(0.3 * len(csv_word))):

                    word.word_error_score = norm_score
                    next_word.word_error_score = norm_score
                    word.error_type = (
                        ErrorType.ERROR if norm_score > 0.5 else ErrorType.NONE
                    )
                    next_word.error_type = (
                        ErrorType.ERROR if norm_score > 0.5 else ErrorType.NONE
                    )
                    page_words_idx += 2
                    csv_idx += 1
                    continue

            # fallback if not a broken hyphenated match
            word.word_error_score = norm_score
            word.error_type = ErrorType.ERROR if norm_score > 0.5 else ErrorType.NONE
            page_words_idx += 1
            csv_idx += 1

        return pred_page
