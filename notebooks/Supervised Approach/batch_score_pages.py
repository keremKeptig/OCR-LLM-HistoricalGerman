#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one pipeline:
1) Word-level labeling (GT vs OCR) -> word_level_labels.jsonl
2) Page-level error rates
3) Feature extraction + RF training -> saved model
4) (Optional) Batch scoring of books -> CSV/JSON/PNGs

Requires your project modules:
  - extract_features.extract_filtered_features_from_page
  - src.evaluator.page_parser.parse_raw_page_from_xml
"""

# Configuration
# Directories for labeling and unlabeled data
GT_DIR = "../../data/d2_0001-0100_with_marginalia"
INPUT_DIR = "../../data/d2_0001-0100_without_marginalia"

# Output paths
WORD_LABELS_JSONL = "../../data/word_level_labels.jsonl"
PAGE_ERROR_RATES_CSV = "../../data/page_error_rates.csv"

# Model + outputs
MODEL_PATH = "./pretrained_weights/rf_regressor.joblib"
OUTDIR = "./outputs"
XML_FOLDERS = "./xml_outputs"

EVAL_FOLDER = "../../data/manual_annotation_approach"

# Optional: books to score after training (each folder = one book with PAGE-XMLs)
SCORE_BOOK_DIRS = [
    "../../data/d2_0001-0100_without_marginalia",
]
# =========================
# ===== END CONFIG ========
# =========================
import sys
from pathlib import Path
import os 

sys.path.append(str(Path(os.getcwd()).parent.parent))

# ---- imports ----
import xml.etree.ElementTree as ET
import re
import json
from difflib import SequenceMatcher
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import traceback
import matplotlib.pyplot as plt

from extract_features import extract_filtered_features_from_page

from src.evaluator.page_parser import parse_raw_page_from_xml, Page, save_pages
from src.evaluator.evaluation import (
    calculate_annotated_pages_with_pred_scorer)


def normalize_text(text):
    # Replace German umlauts and ß with ASCII equivalents
    umlaut_map = {
        'ä': 'a', 'ö': 'o', 'ü': 'u',
        'Ä': 'A', 'Ö': 'O', 'Ü': 'U'
    }
    for orig, repl in umlaut_map.items():
        text = text.replace(orig, repl)
    text = text.replace('\u00A0', ' ')
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize(text):
    text = text.replace('⸗', ' ').replace('-', ' ')
    return re.findall(r'\w+', text)

def line_similarity(line1_words, line2_words):
    if not line1_words or not line2_words:
        return 0.0
    line1_str = ' '.join(line1_words)
    line2_str = ' '.join(line2_words)
    return SequenceMatcher(None, line1_str, line2_str).ratio()


def get_gt_all_lines(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_lines = []
    for tr in root.findall(".//{*}TextRegion"):
        region_type = tr.get("type", "").lower()
        region_id = tr.get("id", None)
        for line in tr.findall(".//{*}TextLine"):
            unicode_el = line.find(".//{*}Unicode")
            if unicode_el is not None and unicode_el.text:
                line_words = tokenize(unicode_el.text.strip())
                line_words_norm = [normalize_text(w) for w in line_words]
                gt_lines.append({
                    'region_type': region_type,
                    'region_id': region_id,
                    'line_id': line.get("id") if line is not None else None,
                    'text': unicode_el.text.strip(),
                    'words': line_words,
                    'words_norm': line_words_norm
                })
    return gt_lines

def get_gt_paragraph_lines(gt_lines):
    paragraphs = []
    para_map = {}
    for gl in gt_lines:
        if gl['region_type'] == "paragraph":
            para_id = gl['region_id']
            if para_id not in para_map:
                para_map[para_id] = []
            para_map[para_id].append(gl)
    for para_id, lines in para_map.items():
        paragraphs.append({'id': para_id, 'lines': lines})
    return paragraphs

def get_input_lines(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for tr in root.findall(".//{*}TextRegion"):
        for line in tr.findall(".//{*}TextLine"):
            unicode_el = line.find(".//{*}Unicode")
            if unicode_el is not None and unicode_el.text:
                line_words = tokenize(unicode_el.text.strip())
                line_words_norm = [normalize_text(w) for w in line_words]
                lines.append({
                    'words': line_words,
                    'words_norm': line_words_norm,
                    'text': unicode_el.text.strip(),
                    'line_id': line.get("id") if line is not None else None,
                    'assigned_para': None,
                    'assigned_gt_line': None,
                    'similarity': 0.0,
                    'matched_to': None,  # can be 'paragraph', 'any_gt', or 'marginalia'
                })
    return lines

def get_marginalia_lines(gt_lines):
    # Return full marginalia GT lines for matching
    return [gl for gl in gt_lines if gl['region_type'] == "marginalia"]

def get_marginalia_phrases(gt_lines):
    phrases = []
    for gl in gt_lines:
        if gl['region_type'] == "marginalia":
            phrase = [normalize_text(w) for w in gl['words']]
            if phrase:
                phrases.append(phrase)
    return phrases

def match_phrase_at(idx, phrase, input_words):
    if idx + len(phrase) > len(input_words):
        return False
    for k in range(len(phrase)):
        if input_words[idx+k] != phrase[k]:
            return False
    return True

def assign_input_lines_to_gt(input_lines, gt_paragraphs, used_gt_line_ids):
    gt_lines_flat = []
    for para in gt_paragraphs:
        for line_idx, line in enumerate(para['lines']):
            if line['line_id'] in used_gt_line_ids:
                continue  # Exclude already used GT lines
            gt_lines_flat.append({
                'para_id': para['id'],
                'line_idx': line_idx,
                'line_data': line
            })
    for input_line in input_lines:
        if input_line['matched_to']:
            continue  # skip if already matched perfectly to marginalia or any_gt
        best_similarity = 0.0
        best_gt_line = None
        for gt_line in gt_lines_flat:
            similarity = line_similarity(input_line['words_norm'], gt_line['line_data']['words_norm'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_gt_line = gt_line
        if best_gt_line and best_similarity > 0:
            input_line['assigned_para'] = best_gt_line['para_id']
            input_line['assigned_gt_line'] = best_gt_line['line_idx']
            input_line['similarity'] = best_similarity
            if best_similarity == 1.0:
                input_line['matched_to'] = 'paragraph'
        else:
            input_line['assigned_para'] = None
            input_line['assigned_gt_line'] = None
            input_line['similarity'] = 0.0

def match_with_context(word_idx, ref_words, ref_used, input_words):
    for ref_ptr in range(len(ref_words)):
        if ref_used[ref_ptr]:
            continue
        if input_words[word_idx] == ref_words[ref_ptr]:
            prev_match = (word_idx > 0 and ref_ptr > 0 and input_words[word_idx-1] == ref_words[ref_ptr-1])
            next_match = (word_idx < len(input_words)-1 and ref_ptr < len(ref_words)-1 and input_words[word_idx+1] == ref_words[ref_ptr+1])
            at_start = (word_idx == 0 or ref_ptr == 0)
            at_end = (word_idx == len(input_words)-1 or ref_ptr == len(ref_words)-1)
            if prev_match or next_match or at_start or at_end:
                return ref_ptr
    return -1

def generate_word_level_labels(gt_dir: Path, input_dir: Path, out_jsonl: Path) -> int:
    output = []
    input_files = sorted(list(input_dir.glob("*.xml")))
    for input_xml in tqdm(input_files, desc="Labeling pages"):
        file_stem = input_xml.stem
        gt_xml = gt_dir / f"{file_stem}.xml"
        if not gt_xml.exists():
            print(f"GT file not found for {file_stem}, skipping")
            continue

        gt_lines = get_gt_all_lines(gt_xml)
        gt_paragraphs = get_gt_paragraph_lines(gt_lines)
        input_lines = get_input_lines(input_xml)
        marginalia_lines = get_marginalia_lines(gt_lines)
        marginalia_phrases = get_marginalia_phrases(gt_lines)

            # --- PHASE 0: Marginalia matching first ---
        marginalia_phrase_set = set(tuple(phrase) for phrase in marginalia_phrases if phrase)
        marginalia_line_ids = set()
        matched_marginalia_phrases = set()
        matched_input_line_ids = set()

        for marg_gt in marginalia_lines:
            marg_tuple = tuple(marg_gt['words_norm'])
            for input_line in input_lines:
                if input_line['matched_to']:
                    continue
                if tuple(input_line['words_norm']) == marg_tuple:
                    input_line['matched_to'] = 'marginalia'
                    matched_marginalia_phrases.add(marg_tuple)
                    matched_input_line_ids.add(input_line['line_id'])
                    marginalia_line_ids.add(input_line['line_id'])
                    break  # Each marginalia GT can be matched to at most one input line

        # --- PHASE 1: Paragraph assignment ---
        used_gt_line_ids = set()
        for input_line in input_lines:
            if input_line['matched_to'] == 'marginalia':
                continue
        assign_input_lines_to_gt(input_lines, gt_paragraphs, used_gt_line_ids)
        for input_line in input_lines:
            if input_line['similarity'] == 1.0 and input_line['assigned_para'] is not None:
                para = next(p for p in gt_paragraphs if p['id'] == input_line['assigned_para'])
                gt_line = para['lines'][input_line['assigned_gt_line']]
                used_gt_line_ids.add(gt_line['line_id'])
                input_line['matched_to'] = 'paragraph'
                matched_input_line_ids.add(input_line['line_id'])

        # --- PHASE 2: Try to match to any GT line for remaining lines ---
        gt_lines_by_wordsnorm = {tuple(gl['words_norm']): gl for gl in gt_lines}
        for input_line in input_lines:
            if input_line['matched_to']:
                continue
            words_norm_tuple = tuple(input_line['words_norm'])
            gt_line = gt_lines_by_wordsnorm.get(words_norm_tuple, None)
            if gt_line and gt_line['line_id'] not in used_gt_line_ids:
                input_line['matched_to'] = 'any_gt'
                used_gt_line_ids.add(gt_line['line_id'])
                matched_input_line_ids.add(input_line['line_id'])

        # --- LABELING ---
        line_wordlabels = {}  # line_id -> list of (word, label)
        para_id_map = {}      # line_id -> para_id

        # PHASE A: Label all perfectly matched input lines as "correct"
        for input_line in input_lines:
            if input_line['matched_to'] in ('marginalia', 'paragraph', 'any_gt'):
                para_id = input_line.get('assigned_para') or input_line.get('para_id') or ""
                para_id_map[input_line['line_id']] = para_id
                line_wordlabels[input_line['line_id']] = [(w, "correct") for w in input_line['words']]

        # PHASE B: Process other input lines by paragraph for context and marginalia-inside-line
        for para in gt_paragraphs:
            para_gt_words = []
            for line in para['lines']:
                if line['line_id'] not in used_gt_line_ids:
                    para_gt_words.extend(line['words_norm'])
            for input_line in [l for l in input_lines if l['assigned_para'] == para['id'] and l['line_id'] not in matched_input_line_ids]:
                input_words = input_line['words']
                input_words_norm = input_line['words_norm']
                input_line_id = input_line['line_id']
                para_id_map[input_line_id] = para['id']
                labels = ["unmatched"] * len(input_words)
                # First: marginalia-inside-line (error marking, skip those matched 100%)
                for phrase in marginalia_phrases:
                    phrase_tuple = tuple(phrase)
                    if phrase_tuple in matched_marginalia_phrases:
                        continue  # Already matched 100%
                    if len(phrase) == 1:
                        for i, word in enumerate(input_words_norm):
                            if labels[i] == "unmatched" and word == phrase[0]:
                                labels[i] = "error"
                    elif len(phrase) > 1:
                        for i in range(len(input_words_norm) - len(phrase) + 1):
                            if all(labels[i+k] == "unmatched" for k in range(len(phrase))) and match_phrase_at(i, phrase, input_words_norm):
                                if input_line_id in marginalia_line_ids:
                                    continue
                                for k in range(len(phrase)):
                                    labels[i+k] = "error"
                # Second: context word matching
                para_gt_used = [False] * len(para_gt_words)
                for i, word in enumerate(input_words_norm):
                    if labels[i] != "unmatched":
                        continue
                    gt_idx = match_with_context(i, para_gt_words, para_gt_used, input_words_norm)
                    if gt_idx != -1:
                        labels[i] = "correct"
                        para_gt_used[gt_idx] = True
                line_wordlabels[input_line_id] = [(w, l) for w, l in zip(input_words, labels)]

        # PHASE C: All other input lines not matched yet
        for input_line in input_lines:
            if input_line['line_id'] in line_wordlabels:
                continue  # Already processed
            input_words = input_line['words']
            input_words_norm = input_line['words_norm']
            input_line_id = input_line['line_id']
            para_id = input_line.get('assigned_para') or input_line.get('para_id') or ""
            para_id_map[input_line_id] = para_id
            if input_line['matched_to'] == 'marginalia':
                labels = ["correct"] * len(input_words)
            else:
                labels = ["unmatched"] * len(input_words)
                for phrase in marginalia_phrases:
                    phrase_tuple = tuple(phrase)
                    if phrase_tuple in matched_marginalia_phrases:
                        continue
                    if len(phrase) == 1:
                        for i, word in enumerate(input_words_norm):
                            if labels[i] == "unmatched" and word == phrase[0]:
                                labels[i] = "error"
                    elif len(phrase) > 1:
                        for i in range(len(input_words_norm) - len(phrase) + 1):
                            if all(labels[i+k] == "unmatched" for k in range(len(phrase))) and match_phrase_at(i, phrase, input_words_norm):
                                if input_line_id in marginalia_line_ids:
                                    continue
                                for k in range(len(phrase)):
                                    labels[i+k] = "error"
            line_wordlabels[input_line_id] = [(w, l) for w, l in zip(input_words, labels)]

        for line_id in line_wordlabels:
            line_wordlabels[line_id] = [(w, "error" if l == "unmatched" else l) for w, l in line_wordlabels[line_id]]

        for in_line in input_lines:
            line_id = in_line['line_id']
            para_id = para_id_map.get(line_id, "")
            for word, label in line_wordlabels[line_id]:
                output.append({
                    "file_id": file_stem,
                    "para_id": para_id,
                    "line_id": line_id,
                    "word": word,
                    "label": label
                })

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(output)} word-level labeled examples to {out_jsonl}")
    return len(output)


def compute_page_error_rates(labels_jsonl: Path, out_csv: Path) -> pd.DataFrame:
    page_labels = defaultdict(list)
    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            page_labels[row["file_id"]].append(row["label"])
    error_rates = {
        file_id: sum(1 for l in labels if l == "error") / len(labels)
        for file_id, labels in page_labels.items() if labels
    }
    df = pd.DataFrame([{"file_id": fid, "error_rate": rate} for fid, rate in error_rates.items()])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved page error rates to {out_csv} ({len(df)} rows).")
    return df


def find_page_xmls(book_dir: Path):
    return sorted([p for p in book_dir.glob("*.xml") if p.is_file()])

def extract_features_from_xml(xml_path: Path, embedder) -> np.ndarray:
    page = parse_raw_page_from_xml(xml_path)
    return extract_filtered_features_from_page(page, embedder)

def train_model_from_error_rates(input_dir: Path,
                                 error_rates_csv: Path,
                                 model_path: Path):
    model_path = Path(model_path)
    if model_path.exists():
        print(f"[INFO] Loading existing model from {model_path}")
        return joblib.load(model_path)

    df_error = pd.read_csv(error_rates_csv)
    xml_files = sorted(list(Path(input_dir).glob("*.xml")))
    df_files = pd.DataFrame({"file_path": [str(f) for f in xml_files], "file_id": [f.stem for f in xml_files]})
    df = df_files.merge(df_error, on="file_id", how="inner").dropna()
    if df.empty:
        raise RuntimeError("No overlap between error-rate pages and INPUT_DIR XMLs.")

    X_paths = df["file_path"].tolist()
    y = df["error_rate"].values

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        X_paths, y, test_size=0.20, random_state=42, shuffle=True
    )

    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def featurize(paths):
        feats = []
        for p in tqdm(paths, desc="Extracting features"):
            try:
                feats.append(extract_features_from_xml(Path(p), embedder))
            except Exception as e:
                sys.stderr.write(f"[WARN] Feature extraction failed for {p}: {e}\n")
                traceback.print_exc(limit=1)
        return np.array(feats)

    X_train = featurize(X_train_paths)
    y_train_f = y_train[:len(X_train)]
    X_test = featurize(X_test_paths)
    y_test_f = y_test[:len(X_test)]

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train_f)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test_f, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_f, y_pred)
    print("Best Params:", grid_search.best_params_)
    print(f"Test MSE : {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")
    return best_model, embedder


def score_books(book_dirs, model, embedder, outdir: Path):
    rows = []
    for bdir in book_dirs:
        bdir = Path(bdir)
        if not bdir.exists() or not bdir.is_dir():
            print(f"[WARN] Skipping non-dir: {bdir}")
            continue
        xmls = find_page_xmls(bdir)
        if not xmls:
            print(f"[WARN] No PAGE-XML in {bdir}")
            continue
        for xml in tqdm(xmls, desc=f"Scoring {bdir.name}", leave=False):
            try:
                feats = extract_features_from_xml(xml, embedder)
                score = float(model.predict([feats])[0])
            except Exception as e:
                sys.stderr.write(f"[WARN] Failed on {xml}: {e}\n")
                traceback.print_exc(limit=1)
                score = np.nan
            rows.append({
                "book_id": bdir.name,
                "page_id": xml.stem,
                "xml_path": str(xml),
                "score": score
            })

    if not rows:
        return pd.DataFrame(columns=["book_id", "page_id", "xml_path", "score"])

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)

    (Path(outdir) / "pagescores.csv").write_text(
        df.sort_values(["book_id", "page_id"]).to_csv(index=False),
        encoding="utf-8"
    )

    with open(Path(outdir) / "pagescores.json", "w", encoding="utf-8") as f:
        grouped = {
            bid: [
                {"page_id": r["page_id"], "score": None if pd.isna(r["score"]) else float(r["score"])}
                for _, r in sub.sort_values("page_id").iterrows()
            ]
            for bid, sub in df.groupby("book_id")
        }
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    # Per-book plots + summary
    summaries = []
    for bid, sub in df.groupby("book_id"):
        # Sort by score (ascending: lower = better)
        sub_sorted = sub.sort_values("score")
        scores   = sub_sorted["score"].astype(float).values
        page_ids = sub_sorted["page_id"].tolist()
        n = len(scores)

        if n:
            # --- Overview plot
            idx = np.arange(n)
            plt.figure(figsize=(16, 4))
            plt.bar(idx, scores, alpha=0.6)
            plt.title(f"{bid}: per-page scores", fontsize=12)
            plt.xlabel("Page ID", fontsize=10)
            plt.ylabel("Predicted error rate", fontsize=10)

            max_labels = 60
            step = max(1, n // max_labels)
            tick_positions = idx[::step]
            tick_labels = [page_ids[i] for i in tick_positions]
            plt.xticks(tick_positions, tick_labels, rotation=60, ha="right", fontsize=7)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.savefig(Path(outdir) / f"{bid}_scores.png", dpi=150)
            plt.close()

        # ----- Summary row with best/worst (min/max) -----
        valid = sub.loc[sub["score"].notna(), ["page_id", "score"]].copy()
        valid["score"] = valid["score"].astype(float)

        if not valid.empty:
            best_idx = valid["score"].idxmin()
            worst_idx = valid["score"].idxmax()
            best_row = valid.loc[best_idx]
            worst_row = valid.loc[worst_idx]

            mean_score = float(valid["score"].mean())
            median_score = float(valid["score"].median())
            best_score = float(best_row["score"])
            best_page_id = str(best_row["page_id"])
            worst_score = float(worst_row["score"])
            worst_page_id = str(worst_row["page_id"])
            num_pages = int(valid.shape[0])
        else:
            mean_score = median_score = best_score = worst_score = float("nan")
            best_page_id = worst_page_id = ""
            num_pages = 0

        summaries.append({
            "book_id": bid,
            "num_pages": num_pages,
            "mean_score": mean_score,
            "median_score": median_score,
            "best_score": best_score,      
            "best_page_id": best_page_id,
            "worst_score": worst_score,     
            "worst_page_id": worst_page_id,
        })

    pd.DataFrame(summaries).sort_values("book_id").to_csv(
        Path(outdir) / "book_summaries.csv", index=False
    )
    print(f"Wrote outputs to {outdir}")
    return df


class FinalRegressorPageScorer:
    def __init__(self, regressor, embedder):
        self.regressor = regressor
        self.embedder = embedder

    def calculate_detailed_error_scores(self, pred_page: Page) -> Page:
        features = extract_filtered_features_from_page(pred_page, self.embedder)
        error_rate = float(self.regressor.predict([features])[0])
        pred_page.overwrite_page_error = error_rate
        return pred_page


def main():
    gt_dir = Path(GT_DIR)
    input_dir = Path(INPUT_DIR)
    labels_jsonl = Path(WORD_LABELS_JSONL)
    outdir = Path(OUTDIR)
    model_path = Path(MODEL_PATH)

    # 1) Word-level labels + page error rates (only if missing)
    if not labels_jsonl.exists():
        print("[INFO] Generating word-level labels...")
        generate_word_level_labels(gt_dir, input_dir, labels_jsonl)
        compute_page_error_rates(labels_jsonl, Path(PAGE_ERROR_RATES_CSV))
    else:
        print(f"[INFO] Found existing labels: {labels_jsonl}")
        if not Path(PAGE_ERROR_RATES_CSV).exists():
            compute_page_error_rates(labels_jsonl, Path(PAGE_ERROR_RATES_CSV))

    # 2) Train (or load) model
    if model_path.exists():
        print(f"[INFO] Loading existing model: {model_path}")
        regressor = joblib.load(model_path)
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    else:
        print("[INFO] Training new model...")
        regressor, embedder = train_model_from_error_rates(
            input_dir=input_dir,
            error_rates_csv=Path(PAGE_ERROR_RATES_CSV),
            model_path=model_path 
                          )
        
    # Dataset path for external/manual annotation
    dataset_path = Path("../../data")
    # Create scorer instance
    pred_scorer = FinalRegressorPageScorer(regressor, embedder)

    # Annotate pages using predicted scorer
    pred_annotated_pages = calculate_annotated_pages_with_pred_scorer(
        dataset_path, pred_scorer
    )

    # Specify output folder for predicted XMLs
    output_folder = Path(XML_FOLDERS)

    # Save the predicted pages as XML files
    written_files = save_pages(pred_annotated_pages, output_folder)

    print(f"Saved {len(written_files)} predicted page XMLs to: {output_folder.resolve()}")

    # 3) Scoring
    if SCORE_BOOK_DIRS:
        score_books(SCORE_BOOK_DIRS, regressor, embedder, outdir)
    else:
        print("[INFO] SCORE_BOOK_DIRS is empty; skipping scoring.")

if __name__ == "__main__":
    main()

