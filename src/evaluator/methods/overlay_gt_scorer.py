from pathlib import Path
from typing import List, Tuple

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageColor, ImageFont
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from src.evaluator.page_parser import *
from src.evaluator.evaluation import ErrorScorerWithGt


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

_OVERLAP_THRESHOLD = 0.30  # IoU ≥ 0.5  ⇒  “high overlap”


def _coords_to_poly(coords: str) -> Polygon:
    pts = [
        (int(x), int(y))
        for x, y in (p.split(",") for p in coords.strip().split() if "," in p)
    ]
    return Polygon(pts).buffer(0) if pts else Polygon()


def _iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union else 0.0


class OverlayGtScorer(ErrorScorerWithGt):
    # ──────────────────────────────────────────────────────────────────────────
    LINE_COVERAGE_THRESHOLD = 0.85
    REGION_IOU_THRESHOLD = 0.30

    def calculate_detailed_error_scores(
        self,
        gt_page: Page,
        pred_page: Page,
    ) -> Page:
        gt_regions: list[tuple[Polygon, RegionType]] = [
            (_coords_to_poly(r.coords), r.type) for r in gt_page.text_regions
        ]

        # --- scoring loop ------------------------------------------------------
        for pred_region in pred_page.text_regions:
            region_poly = _coords_to_poly(pred_region.coords)

            for line in pred_region.lines:
                line_poly = _coords_to_poly(line.coords)
                # Invalid / empty geometry  →  error
                if line_poly.is_empty or not line_poly.is_valid or line_poly.area == 0:
                    line.set_linescore(ErrorType.ERROR, 1.0)
                    continue

                # -------- step 1: find best-covering GT region -----------------
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

                # no GT region covers ≥ 50 % of the line  →  error
                if best_coverage < self.LINE_COVERAGE_THRESHOLD:
                    line.set_linescore(ErrorType.HALLUCINATION, 1.0)
                    continue

                # -------- step 2: paragraph-like GT region ---------------------
                if best_gt_type in _PARAGRAPH_LIKE:
                    line.set_linescore(ErrorType.NONE, (1 - best_coverage) * 4)
                    continue

                # -------- step 3: non-paragraph GT region ----------------------
                if region_poly.is_empty or not region_poly.is_valid:
                    line.set_linescore(ErrorType.ERROR, 1.0)
                    continue

                if _iou(region_poly, best_gt_poly) >= self.REGION_IOU_THRESHOLD:
                    line.set_linescore(ErrorType.NONE, 0.0)
                else:
                    line.set_linescore(ErrorType.ERROR, 1.0)

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
