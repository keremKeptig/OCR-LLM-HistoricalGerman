import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Any
from pagexml.parser import parse_pagexml_file

from src.llms_ocr.common import (
    perplexity,
    get_raw_paragraphs,
    get_word_to_line_mapping,
    chunk_text_by_word,
    remove_newlines,
    replace_historical_chars,
    merge_split_words,
    is_page_number,
)


class ErrorsCalculator:
    """
    Calculates word-level perplexity, identifies problematic regions.
    Args:
        model (Any): The language model used for perplexity calculation.
        tokenizer (Any): The tokenizer corresponding to the model.
        chunk_size (int): Size of text chunks for processing.
        overlap_size (int): Size of overlap between chunks.
    Returns:
        json_data (dict): Contains page-level results including problematic regions.
        df_results (pd.DataFrame): DataFrame with detailed word-level results.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        chunk_size: int = 20,
        overlap_size: int = 15,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ppl_threshold = 158
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.problematic_regions = []
        self.all_word_ppl_sums = []
        self.all_word_ppl_counts = []
        self.all_word_texts = []
        self.all_word_line_ids = []
        self.all_word_paragraph_ids = []
        self.word_smoothed_ppl_page = []

    def process_paragraph(self, paragraph_text_raw: str, paragraph_id: str) -> str:
        """
        Processes a single paragraph, calculates its word-level smoothed perplexity,
        and returns necessary data for page-level aggregation.
        """
        # Preprocess the paragraph text
        merged = merge_split_words(paragraph_text_raw)
        removed_newlines = remove_newlines(merged)
        corrected_paragraph_text = replace_historical_chars(removed_newlines)

        return corrected_paragraph_text

    def process_paragraph_with_line_mapping(
        self, paragraph_text_raw: str, word_line_ids: list, paragraph_id: str
    ):
        """
        Process paragraph text while maintaining line ID mapping through transformations.

        Args:
            paragraph_text_raw: Raw paragraph text
            word_line_ids: List of line IDs for each word in raw text
            paragraph_id: Paragraph identifier

        Returns:
            tuple: (processed_text, updated_line_ids)
        """
        # Start with raw words and their line IDs
        raw_words = paragraph_text_raw.replace("\n", " ").split()

        # Ensure we have line IDs for all words
        if len(word_line_ids) != len(raw_words):
            # Extend or truncate to match
            if len(word_line_ids) < len(raw_words):
                word_line_ids.extend(
                    ["unknown"] * (len(raw_words) - len(word_line_ids))
                )
            else:
                word_line_ids = word_line_ids[: len(raw_words)]

        # Apply text processing steps
        merged = merge_split_words(paragraph_text_raw)
        removed_newlines = remove_newlines(merged)
        corrected_paragraph_text = replace_historical_chars(removed_newlines)

        # Get final processed words
        processed_words = corrected_paragraph_text.split()

        # Simple approach: if word count changed significantly,
        # distribute line IDs proportionally
        if len(processed_words) != len(raw_words):
            # Distribute existing line IDs across new word count
            final_line_ids = []
            for i, processed_word in enumerate(processed_words):
                # Map to closest original word position
                original_idx = min(i, len(word_line_ids) - 1)
                final_line_ids.append(word_line_ids[original_idx])
        else:
            # Same number of words, keep original mapping
            final_line_ids = word_line_ids

        return corrected_paragraph_text, final_line_ids

    def get_words_per_paragraph_metrics(self, par_text: str):
        """
        Calculates word-level perplexity for a paragraph, returning smoothed perplexity values.

        Args:
            par_text (str): The text of the paragraph to analyze.

        Returns:
            tuple: A tuple containing:
                - paragraph_words (list): List of words in the paragraph.
                - word_ppl_sums (np.ndarray): Array of perplexity sums for each word.
                - word_ppl_counts (np.ndarray): Array of counts for each word.
        """

        paragraph_words = par_text.split()
        total_words_in_paragraph = len(paragraph_words)

        # Get word chunks and calculate perplexity/error for the current paragraph
        words_chunks = chunk_text_by_word(
            text=par_text,
            chunk_size=self.chunk_size,
            overlap_size=self.overlap_size,
        )
        df_para_chunks = pd.DataFrame(words_chunks)

        df_para_chunks["ppl"] = df_para_chunks["text"].apply(
            lambda t: perplexity(t, self.model, self.tokenizer)
        )

        word_ppl_sums_para = pd.Series(0.0, index=range(total_words_in_paragraph))
        word_ppl_counts_para = pd.Series(0, index=range(total_words_in_paragraph))

        for _, row in df_para_chunks.iterrows():
            start = row["start_word_id"]
            end = row["end_word_id"]
            ppl = row["ppl"]
            word_ppl_sums_para.loc[start:end] += ppl
            word_ppl_counts_para.loc[start:end] += 1

        word_smoothed_ppl_paragraph = word_ppl_sums_para / word_ppl_counts_para
        word_smoothed_ppl_paragraph = word_smoothed_ppl_paragraph.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        paragraph_mean_ppl_smoothed = word_smoothed_ppl_paragraph[
            word_smoothed_ppl_paragraph > 0
        ].mean()
        if pd.isna(paragraph_mean_ppl_smoothed):
            paragraph_mean_ppl_smoothed = 0.0

        return paragraph_words, word_ppl_sums_para.values, word_ppl_counts_para.values

    def process_page(self, xml_file: Path):
        """
        Processes a single page of XML, extracting and analyzing text data.

        Args:
            xml_file (Path): The path to the XML file to process.
        """
        base_name = xml_file.stem
        self.problematic_regions = []
        self.all_word_ppl_sums = []
        self.all_word_ppl_counts = []
        self.all_word_texts = []
        self.all_word_line_ids = []
        self.all_word_paragraph_ids = []  # Reset paragraph IDs tracking
        self.word_smoothed_ppl_page = []

        print(f"Processing page: {base_name}")
        page = parse_pagexml_file(xml_file)
        raw_paragraphs = get_raw_paragraphs(xml_doc=page)
        word_line_mappings = get_word_to_line_mapping(xml_doc=page)

        for para_idx in tqdm(
            raw_paragraphs, desc=f"Processing {base_name} paragraphs", disable=True
        ):

            # handle empty paragraphs
            if raw_paragraphs[para_idx] == "":
                continue
            # page number has its own paragraph => correct
            elif is_page_number(raw_paragraphs[para_idx]):
                # Process page number but skip perplexity calculation
                processed_para, final_line_ids = (
                    self.process_paragraph_with_line_mapping(
                        paragraph_text_raw=raw_paragraphs[para_idx],
                        word_line_ids=word_line_mappings[para_idx],
                        paragraph_id=para_idx,
                    )
                )

                # Add page number words to lists
                page_words = processed_para.split()
                self.all_word_texts.extend(page_words)
                self.all_word_line_ids.extend(final_line_ids[: len(page_words)])
                self.all_word_paragraph_ids.extend([para_idx] * len(page_words))

                # Add zero perplexity values (page numbers are not errors)
                zero_ppl_sums = np.zeros(len(page_words))
                zero_ppl_counts = np.ones(
                    len(page_words)
                )  # Use 1 to avoid division by zero
                self.all_word_ppl_sums.append(zero_ppl_sums)
                self.all_word_ppl_counts.append(zero_ppl_counts)
                continue

            # Process paragraph while preserving line mapping
            processed_para, final_line_ids = self.process_paragraph_with_line_mapping(
                paragraph_text_raw=raw_paragraphs[para_idx],
                word_line_ids=word_line_mappings[para_idx],
                paragraph_id=para_idx,
            )

            p_words, p_ppl_sums, p_ppl_counts = self.get_words_per_paragraph_metrics(
                processed_para
            )
            self.all_word_texts.extend(p_words)
            self.all_word_line_ids.extend(final_line_ids)
            self.all_word_paragraph_ids.extend([para_idx] * len(p_words))
            self.all_word_ppl_sums.append(p_ppl_sums)
            self.all_word_ppl_counts.append(p_ppl_counts)

        if not self.all_word_ppl_sums or not self.all_word_ppl_counts:
            json_data = {
                "page_id": base_name,
                "all_words_page": [],
                "all_word_line_ids": [],
                "word_smoothed_ppl_page": [],
                "problematic_regions": [],
            }
            # Create empty DataFrame
            df_results = pd.DataFrame()
            return json_data, df_results
        else:
            combined_ppl_sums = np.concatenate(self.all_word_ppl_sums)
            combined_ppl_counts = np.concatenate(self.all_word_ppl_counts)

            self.word_smoothed_ppl_page = np.divide(
                combined_ppl_sums,
                combined_ppl_counts,
                where=combined_ppl_counts != 0,  # Condition to avoid division by zero
            )
            self.word_smoothed_ppl_page = np.nan_to_num(
                self.word_smoothed_ppl_page, nan=0.0, posinf=0.0, neginf=0.0
            )

            error_flags_page = self.word_smoothed_ppl_page > self.ppl_threshold

            current_part_start_idx = -1

            for i, is_error in enumerate(error_flags_page):
                if is_error and current_part_start_idx == -1:
                    # Start of a new problematic part
                    current_part_start_idx = i
                elif (
                    not is_error and current_part_start_idx != -1
                ):  # end of the problematic part
                    self.problematic_regions.append(
                        {
                            "type": "full_text",
                            "start_idx": current_part_start_idx,
                            "end_idx": i - 1,
                            "words": self.all_word_texts[current_part_start_idx:i],
                            "line_ids": self.all_word_line_ids[
                                current_part_start_idx:i
                            ],
                        }
                    )
                    current_part_start_idx = -1

            # Check for a problematic part ending at the very end of the page
            if current_part_start_idx != -1:
                self.problematic_regions.append(
                    {
                        "type": "full_text",
                        "start_idx": current_part_start_idx,
                        "end_idx": len(error_flags_page) - 1,
                        "words": self.all_word_texts[current_part_start_idx:],
                        "line_ids": self.all_word_line_ids[current_part_start_idx:],
                    }
                )

            json_data = {
                "page_id": base_name,
                "all_words_page": self.all_word_texts,
                "all_word_line_ids": self.all_word_line_ids,
                "word_smoothed_ppl_page": self.word_smoothed_ppl_page.tolist(),
                "problematic_regions": self.problematic_regions,
            }

        # Create DataFrame with detailed word information
        df_results = self.create_results_dataframe(base_name)

        return json_data, df_results

    def save_json(self, data, out_dir, file_name="results.json"):
        """
        Saves the results of the processing to a JSON file.
        Args:
            data (dict): The data to save.
            out_dir (str): The directory where the JSON file will be saved.
            file_name (str): The name of the JSON file.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def save_dataframe(self, df, out_dir, file_name="results.csv"):
        """
        Saves the results DataFrame to a CSV file.
        Args:
            df (pd.DataFrame): The DataFrame to save.
            out_dir (str): The directory where the CSV file will be saved.
            file_name (str): The name of the CSV file.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        df.to_csv(os.path.join(out_dir, file_name), index=False, encoding="utf-8")

    def create_results_dataframe(self, base_name):
        """Creates a DataFrame with detailed results for each word on the page.

        Args:
            base_name (str): The base name of the page (used for the page_id).

        Returns:
            pd.DataFrame: A DataFrame containing detailed results with columns:
            - word: The text of the word.
            - line_id: The ID of the line (from original xml) the word is on.
            - paragraph_id: The ID of the paragraph (from original xml) the word is in.
            - word_position: The position of the word in the text.
            - perplexity: The smoothed perplexity score for the word.
            - is_error: Whether the word is considered an error based on its perplexity.
            - page_id: The ID of the page the word is on.
        """

        if not self.all_word_texts:
            return pd.DataFrame()

        # Calculate final perplexity scores
        if self.all_word_ppl_sums and self.all_word_ppl_counts:
            combined_ppl_sums = np.concatenate(self.all_word_ppl_sums)
            combined_ppl_counts = np.concatenate(self.all_word_ppl_counts)

            word_smoothed_ppl = np.divide(
                combined_ppl_sums,
                combined_ppl_counts,
                where=combined_ppl_counts != 0,
            )
            word_smoothed_ppl = np.nan_to_num(
                word_smoothed_ppl, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            word_smoothed_ppl = np.zeros(len(self.all_word_texts))

        # Create DataFrame
        df = pd.DataFrame(
            {
                "word": self.all_word_texts,
                "line_id": self.all_word_line_ids,
                "paragraph_id": (
                    self.all_word_paragraph_ids
                    if hasattr(self, "all_word_paragraph_ids")
                    else ["unknown"] * len(self.all_word_texts)
                ),
                "word_position": range(len(self.all_word_texts)),
                "perplexity": word_smoothed_ppl,
                "is_error": word_smoothed_ppl > self.ppl_threshold,
                "page_id": base_name,
            }
        )

        return df
