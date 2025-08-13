from pathlib import Path
from typing import List, Tuple

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageColor, ImageFont
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

from src.evaluator.evaluation import ErrorScorerWithGt


class OverlayGtScorer(ErrorScorerWithGt):
    """Compare PAGE‑XML text regions against ground‑truth and create an overlay.

    Matching rules
    -------------
    * **Non‑paragraph** GT regions must correspond to **exactly one** predicted
      text‑region with sufficient IoU.
    * **Paragraph** GT regions may be fragmented in the prediction; all
      predictions that overlap contribute to the combined coverage.

    The method returns a single **error score** (0 … 100, lower is better).
    The score is also visualised together with GT/predicted polygons and the
    intersection quality (OK / warning / error) if *WRITE_OVERLAY* is set.
    """

    # ──────────────────────────── XML helpers ────────────────────────────
    @staticmethod
    def _parse_pagexml_polygons(xml_path: Path) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        ns = {"pg": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}
        root = ET.parse(xml_path).getroot()
        polygons, types = [], []
        for region in root.findall(".//pg:TextRegion", ns):
            region_type = region.attrib.get("type", "").lower()
            coords = region.find("pg:Coords", ns)
            if coords is None:
                continue
            pts = [tuple(map(int, p.split(","))) for p in coords.attrib["points"].split() if "," in p]
            if pts:
                polygons.append(pts)
                types.append(region_type)
        return polygons, types

    # ───────────────────────────── geometry ─────────────────────────────
    @staticmethod
    def _iou(a: Polygon, b: Polygon) -> float:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union else 0.0

    @staticmethod
    def _geom_to_polys(geom) -> List[List[Tuple[int, int]]]:
        """Convert shapely geometry to a list of *pixel* polygons (x, y)."""
        if geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            return [[(int(x), int(y)) for x, y in geom.exterior.coords]]
        if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            polys: List[List[Tuple[int, int]]] = []
            for g in geom.geoms:
                polys.extend(OverlayGtScorer._geom_to_polys(g))
            return polys
        return []

    # ────────────────────────── drawing helpers ─────────────────────────
    @staticmethod
    def _fill_single_layer(img: Image.Image, polys: List[List[Tuple[int, int]]], colour: str, opacity: int = 80) -> None:
        if not polys:
            return
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        rgba = ImageColor.getrgb(colour) + (opacity,)
        for p in polys:
            draw.polygon(p, fill=rgba)
        img.alpha_composite(layer)

    # ─────────────────────────── main entry ─────────────────────────────
    def calculate_page_error_score(
        self,
        gt_image_path: Path,
        gt_xml_path: Path,
        pred_image_path: Path,
        pred_xml_path: Path,
    ) -> float:
        ERROR_TH = 0.33
        WARNING_TH = 0.5
        WRITE_OVERLAY = False
        SCORE_FOR_ERROR = 1
        SCORE_FOR_WARNING = 0.5
        SCORE_FOR_MISSING = 0.5

        # ---------- load data ----------
        gt_pts, gt_types = self._parse_pagexml_polygons(gt_xml_path)
        pred_pts, _ = self._parse_pagexml_polygons(pred_xml_path)

        gt_polys = [Polygon(p).buffer(0) for p in gt_pts]
        pred_polys = [Polygon(p).buffer(0) for p in pred_pts]
        pred_used = [False] * len(pred_polys)

        overlap_ok: List[List[Tuple[int, int]]] = []
        overlap_warn: List[List[Tuple[int, int]]] = []
        overlap_err: List[List[Tuple[int, int]]] = []
        score = 0.0

        # ------------------------------------------------------------------
        # iterate over ground‑truth regions
        # ------------------------------------------------------------------
        for g_idx, g_poly in enumerate(gt_polys):
            g_type = gt_types[g_idx]

            if g_type == "paragraph":
                # ----------------------------------------------------------
                # paragraph must be *mostly* covered by **at least one**
                # predicted region (it can be the same big region for many
                # paragraphs).  We therefore measure the fraction of the GT
                # paragraph that lies inside each overlapping prediction and
                # keep the *best* coverage.
                # ----------------------------------------------------------
                best_cov, best_inter, best_p_idx = 0.0, None, -1
                for p_idx, p_poly in enumerate(pred_polys):
                    inter = g_poly.intersection(p_poly)
                    if inter.is_empty:
                        continue
                    coverage = inter.area / g_poly.area if g_poly.area else 0.0
                    if coverage > best_cov:
                        best_cov, best_inter, best_p_idx = coverage, inter, p_idx

                if best_p_idx == -1:
                    # nothing overlaps → complete miss
                    overlap_err += self._geom_to_polys(g_poly)
                    score += SCORE_FOR_MISSING
                    continue

                bucket = (
                    overlap_err if best_cov < ERROR_TH else
                    overlap_warn if best_cov < WARNING_TH else
                    overlap_ok
                )
                bucket.extend(self._geom_to_polys(best_inter))

                score += SCORE_FOR_ERROR if best_cov < ERROR_TH else SCORE_FOR_WARNING if best_cov < WARNING_TH else 0
                pred_used[best_p_idx] = True

            else:
                # ----------------------------------------------------------
                # non‑paragraph → must match exactly one predicted region
                # ----------------------------------------------------------
                best_iou, best_p_idx = 0.0, -1
                for p_idx, p_poly in enumerate(pred_polys):
                    if pred_used[p_idx]:
                        continue
                    iou = self._iou(g_poly, p_poly)
                    if iou > best_iou:
                        best_iou, best_p_idx = iou, p_idx

                if best_p_idx == -1:
                    overlap_err += self._geom_to_polys(g_poly)
                    score += 1
                    continue

                # ensure region is not split across multiple predictions
                duplicate_match = any(
                    (not pred_used[p_idx]) and p_idx != best_p_idx and self._iou(g_poly, pred_polys[p_idx]) >= ERROR_TH
                    for p_idx in range(len(pred_polys))
                )

                if duplicate_match:
                    overlap_err += self._geom_to_polys(g_poly)
                    score += 1
                else:
                    inter = g_poly.intersection(pred_polys[best_p_idx])
                    bucket = (
                        overlap_err if best_iou < ERROR_TH else
                        overlap_warn if best_iou < WARNING_TH else
                        overlap_ok
                    )
                    bucket.extend(self._geom_to_polys(inter))
                    score += 1 if best_iou < ERROR_TH else 0.5 if best_iou < WARNING_TH else 0

                pred_used[best_p_idx] = True

        # ------------------------------------------------------------------
        # unmatched predictions count as error
        # ------------------------------------------------------------------
        for p_idx, used in enumerate(pred_used):
            if not used:
                overlap_err += self._geom_to_polys(pred_polys[p_idx])
                score += 1

        score = min(score, 100)

        # ------------------------------------------------------------------
        # visualisation overlay (optional)
        # ------------------------------------------------------------------
        if WRITE_OVERLAY:
            img_folder = gt_image_path.parent
            output_dir = img_folder.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = gt_image_path.stem

            # simple overlay (GT-paragraphs orange, other GT purple, predictions turquoise)
            simple = Image.open(gt_image_path).convert("RGBA")
            para_pts = [p for p, t in zip(gt_pts, gt_types) if t == "paragraph"]
            other_pts = [p for p, t in zip(gt_pts, gt_types) if t != "paragraph"]

            PARA_COL, OTHER_COL, PRED_COL = "orange", "mediumpurple", "turquoise"
            self._fill_single_layer(simple, para_pts, PARA_COL, 80)
            self._fill_single_layer(simple, other_pts, OTHER_COL, 80)
            self._fill_single_layer(simple, pred_pts, PRED_COL, 80)

            # add legend at the bottom-left
            draw_simple = ImageDraw.Draw(simple)
            try:
                font_simple = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            except IOError:
                font_simple = ImageFont.load_default()

            legend_simple = [
                ("GT-Paragraph", PARA_COL),
                ("GT-Marginalia", OTHER_COL),
                ("Kraken Overlay", PRED_COL),
            ]
            bw_s, bh_s, gap_s, mar_s = 30, 20, 6, 15
            get_w_s = (
                lambda t: font_simple.getlength(t)
                if hasattr(font_simple, "getlength")
                else font_simple.getsize(t)[0]
            )
            panel_w_s = bw_s + 8 + max(get_w_s(l) for l, _ in legend_simple)
            panel_h_s = len(legend_simple) * bh_s + (len(legend_simple) - 1) * gap_s
            xs0, ys0 = mar_s, simple.height - panel_h_s - mar_s
            ImageDraw.Draw(simple, "RGBA").rectangle(
                [xs0 - 6, ys0 - 6, xs0 + panel_w_s + 6, ys0 + panel_h_s + 6],
                fill=(0, 0, 0, 120),
            )
            ys = ys0
            for lbl, col in legend_simple:
                ImageDraw.Draw(simple).rectangle(
                    [xs0, ys, xs0 + bw_s, ys + bh_s], fill=col, outline="black"
                )
                ths = (
                    font_simple.getsize(lbl)[1]
                    if hasattr(font_simple, "getsize")
                    else bh_s
                )
                draw_simple.text((xs0 + bw_s + 8, ys + (bh_s - ths) // 2), lbl, "white", font_simple)
                ys += bh_s + gap_s

            simple_out = output_dir / f"{base_name}.overlay_simple.png"
            simple.save(simple_out)
            print(f"Simple overlay written to {simple_out}")
            # detailed overlay (blue GT, grey pred, coloured intersections)
            img = Image.open(gt_image_path).convert("RGBA")
            self._fill_single_layer(img, gt_pts, "blue", 20)
            self._fill_single_layer(img, pred_pts, "grey", 20)
            self._fill_single_layer(img, overlap_ok, "green", 100)
            self._fill_single_layer(img, overlap_warn, "yellow", 100)
            self._fill_single_layer(img, overlap_err, "red", 100)

            draw = ImageDraw.Draw(img)
            try:
                font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
            except IOError:
                font_big = ImageFont.load_default()

            legend = [
                ("GT", "blue"),
                ("Pred", "grey"),
                ("OK", "green"),
                ("Warning", "yellow"),
                ("Error", "red"),
                (f"Score: {score:.0f}", "white"),
            ]
            bw, bh, gap, mar = 36, 24, 8, 20
            get_w = (lambda t: font_big.getlength(t) if hasattr(font_big, "getlength") else font_big.getsize(t)[0])
            panel_w = bw + 10 + max(get_w(l) for l, _ in legend)
            panel_h = len(legend) * bh + (len(legend) - 1) * gap
            x0, y0 = img.width - panel_w - mar, img.height - panel_h - mar
            ImageDraw.Draw(img, "RGBA").rectangle([x0 - 6, y0 - 6, x0 + panel_w + 6, y0 + panel_h + 6], fill=(0, 0, 0, 120))
            y = y0
            for lbl, col in legend:
                ImageDraw.Draw(img).rectangle([x0, y, x0 + bw, y + bh], fill=col, outline="black")
                th = font_big.getsize(lbl)[1] if hasattr(font_big, "getsize") else bh
                draw.text((x0 + bw + 10, y + (bh - th) // 2), lbl, "white", font_big)
                y += bh + gap

            full_out = output_dir / f"{base_name}.overlay.png"
            img.save(full_out)
            print(f"Overlay image written to {full_out}")

        return score
