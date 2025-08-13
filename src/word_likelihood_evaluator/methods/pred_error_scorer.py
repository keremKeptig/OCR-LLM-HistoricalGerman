import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from fuzzywuzzy import fuzz, process

from src.word_likelihood_evaluator.evaluation import ErrorScorerFromPred

def normalize_text(text):
    """Cleans and normalizes a string for reliable comparison."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('⸗', '-').replace('–', '-').replace('—', '-')
    text = text.replace('\u00A0', ' ')
    return " ".join(text.split())

def load_predicted_errors(csv_filepath: Path):
    """Loads predicted OCR errors from a CSV file."""
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            return pd.read_csv(f).to_dict('records')
    except FileNotFoundError:
        return None

def parse_gold_to_word_list(xml_filepath: Path):
    """
    Parses the XML and returns a flat list of every word, tagged with its
    region type, text, and its index to find neighbors.
    """
    # empty‐file check
    try:
        if xml_filepath.stat().st_size == 0:
            return []
    except FileNotFoundError:
        return None

    try:
        tree = ET.parse(xml_filepath)
    except ET.ParseError as e:
        return []
    root = tree.getroot()

    namespaces = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}
    all_words = []
    for region in root.findall('.//pc:TextRegion', namespaces):
        region_type = region.get('type')
        # concat all lines in region
        txt = " ".join(
            u.text for u in region.findall('.//pc:Unicode', namespaces) if u.text
        )
        words = normalize_text(txt).split()
        for idx, w in enumerate(words):
            all_words.append({
                'text':        w,
                'type':        region_type,
                'index':       idx,
                'region_words': words
            })
    return all_words

def assess_predictions(predicted_errors, gold_word_list):
    """
    Assesses predictions using paragraph-first priority and fuzzy context matching on all plausible matches.
    """
    true_positives = 0
    false_positives = 0

    ERROR_TYPES = {"marginalia", "footnote", "page-number", "header", "decoration", "other"}
    WORD_SIMILARITY_THRESHOLD = 60
    NEIGHBOR_SIMILARITY_THRESHOLD = 60

    paragraph_words = [word for word in gold_word_list if word['type'] == 'paragraph']
    error_words = [word for word in gold_word_list if word['type'] in ERROR_TYPES]

    unique_paragraph_word_choices = list({w['text'] for w in paragraph_words})
    unique_error_word_choices     = list({w['text'] for w in error_words})

    for pred in predicted_errors:
        pred_word = normalize_text(pred['word'])
        pred_context_words = normalize_text(pred['context']).split()

        if not pred_word:
            continue
        was_handled = False

        if unique_paragraph_word_choices:
            # find all plausible matches then check neighbors for each match
            plausible_matches = process.extractBests(
                pred_word,
                unique_paragraph_word_choices,
                scorer=fuzz.ratio,
                score_cutoff=WORD_SIMILARITY_THRESHOLD
            )

            is_confirmed_fp = False
            for match_word, score in plausible_matches:
                instances = [w for w in paragraph_words if w['text'] == match_word]
                for inst in instances:
                    # prediction neighbors
                    pred_prev = pred_next = None
                    if pred_context_words:
                        try:
                            anchor, _ = process.extractOne(pred_word, pred_context_words)
                            idx = pred_context_words.index(anchor)
                            pred_prev = pred_context_words[idx-1] if idx>0 else None
                            pred_next = pred_context_words[idx+1] if idx<len(pred_context_words)-1 else None
                        except (ValueError, IndexError):
                            pass

                    # gold neighbors
                    gi = inst['index']
                    gw = inst['region_words']
                    gold_prev = gw[gi-1] if gi>0 else None
                    gold_next = gw[gi+1] if gi<len(gw)-1 else None

                    prev_match = (pred_prev and gold_prev and
                                  fuzz.ratio(pred_prev, gold_prev) > NEIGHBOR_SIMILARITY_THRESHOLD)
                    next_match = (pred_next and gold_next and
                                  fuzz.ratio(pred_next, gold_next) > NEIGHBOR_SIMILARITY_THRESHOLD)

                    if prev_match or next_match:
                        is_confirmed_fp = True
                        break
                if is_confirmed_fp:
                    break

            if plausible_matches:
                if is_confirmed_fp:
                    false_positives += 1
                else:
                    true_positives += 1
                was_handled = True

        if not was_handled and unique_error_word_choices:
            best_match_err, score_err = process.extractOne(pred_word, unique_error_word_choices, scorer=fuzz.ratio)
            if score_err >= WORD_SIMILARITY_THRESHOLD:
                true_positives += 1
                was_handled = True

        if not was_handled:
            true_positives += 1

    total_actual_error_words = len(error_words)
    false_negatives = max(0, total_actual_error_words - true_positives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall    = true_positives / (true_positives + false_negatives)  if (true_positives + false_negatives) else 0
    f1_score  = (2 * precision * recall) / (precision + recall)         if (precision + recall) else 0

    return {
        'true_positives':  true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision':       precision,
        'recall':          recall,
        'f1_score':        f1_score
    }


class PredictionErrorScorer(ErrorScorerFromPred):
    
    def __init__(self, csv_folder: Path, xml_folder: Path):
        self.csv_folder = csv_folder
        self.xml_folder = xml_folder

    def calculate_page_error_score(self, image_path: Path, xml_path: Path) -> float:
        stem = xml_path.stem
        csv_path = self.csv_folder / f"{stem}_ocr_errors.csv"
        preds = load_predicted_errors(csv_path)
        if preds is None:
            return 100.0

        gold_list = parse_gold_to_word_list(self.xml_folder / f"{stem}.xml")
        if gold_list is None:
            return 100.0
        stats = assess_predictions(preds, gold_list)
        return (1.0 - stats['f1_score'])*100.0
