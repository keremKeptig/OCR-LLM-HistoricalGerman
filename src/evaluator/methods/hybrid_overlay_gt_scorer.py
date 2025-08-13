from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageColor, ImageFont
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from src.evaluator.page_parser import *
from src.evaluator.evaluation import ErrorScorerWithGt

import re
from difflib import SequenceMatcher

_PARAGRAPH_LIKE: set[RegionType] = {
    RegionType.PARAGRAPH,
    RegionType.HEADER,
    RegionType.HEADING,
    RegionType.FOOTNOTE,
}
_NON_PARAGRAPH: set[RegionType] = {
    RegionType.MARGINALIA,
    RegionType.PAGE_NUMBER,
    RegionType.OTHER,
}

class HybridOverlayGtScorer(ErrorScorerWithGt):
    OVERLAP_THRESHOLD = 0.30
    LINE_COVERAGE_THRESHOLD = 0.85
    REGION_IOU_THRESHOLD = 0.30
    LINE_SIMILARITY_THRESHOLD = 0.53;
    def calculate_detailed_error_scores(
        self,
        gt_page: Page,
        pred_page: Page,
    ) -> Page:
        """Score *pred_page* w.r.t *gt_page* using geometry first, then text."""
        # ── Index all GT regions/polygons once  ───────────────────────────
        gt_regions: list[tuple[Polygon, RegionType]] = [
            (_coords_to_poly(r.coords), r.type) for r in gt_page.text_regions
        ]

        # Add a fast lookup table *by line* for lexical matching
        gt_line_index: list[tuple[Polygon, list[str]]] = []
        for r in gt_page.text_regions:
            for l in r.lines:
                gt_line_index.append(
                    (
                        _coords_to_poly(l.coords),
                        _tokenize(_normalize_text(str(l))),
                    )
                )

        for pred_region in pred_page.text_regions:
            region_poly = _coords_to_poly(pred_region.coords)
            prev_line = ""
            for line in pred_region.lines:
                if 'den Perserkriegen hatten' in str(line):
                    line = line
                line_poly = _coords_to_poly(line.coords)

                # 1. Geometry sanity check --------------------------------
                if (line_poly.is_empty or not line_poly.is_valid or line_poly.area == 0):
                    line.set_linescore(ErrorType.ERROR, 1.0)
                    continue

                # 2. Best covering GT region ------------------------------
                best_gt_poly: Polygon | None = None
                best_gt_type: RegionType | None = None
                best_coverage = 0.0

                for gt_poly, gt_type in gt_regions:
                    if gt_poly.is_empty:
                        continue
                    inter_area = line_poly.intersection(gt_poly).area
                    coverage = inter_area / line_poly.area
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_gt_poly = gt_poly
                        best_gt_type = gt_type

                # 2a. Not covered → hallucination ------------------------
                if best_coverage < self.LINE_COVERAGE_THRESHOLD:
                    line.set_linescore(ErrorType.HALLUCINATION, 1.0)
                    continue

                # 3. Paragraph‑like regions? apply lexical check ----------
                if best_gt_type.value in {r.value for r in _PARAGRAPH_LIKE}:


                    # Collect and normalize+tokenize all text from the matching GT region
                    gt_tokens: list[str] = []
                    for gt_region in gt_page.text_regions:
                        poly = _coords_to_poly(gt_region.coords)
                        if poly.equals(best_gt_poly):
                            for gt_line in gt_region.lines:
                                gt_tokens.extend(_tokenize(_normalize_text(str(gt_line))))
                            # Normalize and tokenize predicted prev + line (or only first if first poly of gt)
                            if _coords_to_poly(gt_region.lines[0].coords).equals(poly): # this does not yet work
                                pred_tokens = _tokenize(_normalize_text(str(line)))
                            else:
                                pred_tokens = _tokenize(_normalize_text(str(prev_line) + str(line)))
                            break
                    # 3. Compute similarity score between predicted line and GT region text
                    similarity = partial_ratio(pred_tokens, gt_tokens)

                    # 4. Threshold-based error setting
                    if similarity < self.LINE_SIMILARITY_THRESHOLD:  # You can tune this threshold
                        line.set_linescore(ErrorType.MERGE, 1.0)
                    else:
                        line.set_linescore(ErrorType.NONE, (1 - best_coverage)*4)
                    continue

                # 4. Region sanity (non‑paragraph path) -------------------
                if region_poly.is_empty or not region_poly.is_valid:
                    line.set_linescore(ErrorType.ERROR, 1.0)
                    continue

                # 5. Non‑paragraph strict IoU -----------------------------
                if _iou(region_poly, best_gt_poly) >= self.REGION_IOU_THRESHOLD:
                    line.set_linescore(ErrorType.NONE, 0.0)
                else:
                    line.set_linescore(ErrorType.ERROR, 1.0)

                prev_line = line
        return pred_page


def _best_iou(poly: Polygon, gt_polys: Iterable[Polygon]) -> float:
    return max((_iou(poly, p) for p in gt_polys), default=0.0)

def _set_line_score(
    line: TextLine,
    error_type: ErrorType,
    score: float,
) -> None:
    for w in line.words:
        w.word_error_score = score
        w.error_type = error_type
_umlaut_map = {
    "ä": "a", "ö": "o", "ü": "u",
    "Ä": "A", "Ö": "O", "Ü": "U",
    "ß": "ss",
}


def _normalize_text(text: str) -> str:
    """Lower‑case, collapse whitespace, replace German umlauts/ß, strip."""
    for orig, repl in _umlaut_map.items():
        text = text.replace(orig, repl)
    text = text.replace("\u00A0", " ")  # non‑breaking space
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer that breaks on punctuation and hyphens."""
    text = text.replace("⸗", " ").replace("-", " ")
    return re.findall(r"\w+", text)


def _line_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    """Levenshtein‑ratio over the *string* (cheap but works well enough)."""
    if not tokens1 or not tokens2:
        return 0.0
    return SequenceMatcher(None, " ".join(tokens1), " ".join(tokens2)).ratio()

def partial_ratio(a: str, b: str) -> float:
    """Computes fuzzy partial match ratio."""
    if len(a) < len(b):
        a, b = b, a  # Ensure 'a' is the longer string

    max_ratio = 0.0
    len_b = len(b)
    for i in range(len(a) - len_b + 1):
        window = a[i:i+len_b]
        ratio = SequenceMatcher(None, window, b).ratio()
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def _coords_to_poly(coords: str) -> Polygon:
    pts = [
        (int(x), int(y))
        for x, y in (
            p.split(",") for p in coords.strip().split() if "," in p
        )
    ]
    return Polygon(pts).buffer(0) if pts else Polygon()


def _iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union else 0.0