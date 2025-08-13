from __future__ import annotations
import xml.etree.ElementTree as ET  # ✅ Needed for ET.parse
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Iterable, Tuple  # ✅ Tuple needed for return type
from PIL import Image, ImageDraw, ImageFont  # ⬅ typing + runtime use
from numpy.f2py.auxfuncs import throw_error


class RegionType(Enum):
    PARAGRAPH = auto()
    PAGE_NUMBER = auto()
    HEADER = auto()
    HEADING = auto()
    MARGINALIA = auto()
    FOOTNOTE = auto()
    OTHER = auto()


class ErrorType(Enum):
    NONE = auto()
    OCR = auto()
    SPELLING = auto()
    HALLUCINATION =auto()
    MISSING = auto()
    ERROR = auto()
    MERGE = auto()

@dataclass
class Page:
    xml_path: Path
    name: str = field(init=False)
    overwrite_page_error:float = -1.0 # This can overwrite the page error for approaches that dont use word level errors (e.g. manual)
    imageHeight: int = 0
    imageWidth: int = 0
    image_path: Optional[Path] = None
    text_regions: List[TextRegion] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = self.xml_path.stem

    def iter_words(self) -> Iterable[Word]:
        for region in self.text_regions:
            for line in region.lines:
                yield from line.words

    def iter_words_with_line_id(self) -> Iterable[tuple[str, Word]]:
        for region in self.text_regions:
            for line in region.lines:
                for word in line.words:
                    yield (region.id, line.id, word)

    def get_amount_of_error_vs_total(self) -> Tuple[int, int]:
        total = 0
        errors = 0
        for w in self.iter_words():
            total += 1
            if w.error_type.value != ErrorType.NONE.value:
                errors += 1
        return errors, total

    def calculate_total_relative_error_score(self) -> float:
        if (self.overwrite_page_error != -1):
            return self.overwrite_page_error # overwrite if set manually
        errors, total = self.get_amount_of_error_vs_total()
        return errors / total if total else 0.0

@dataclass
class TextRegion:
    coords: str
    id: str
    type: RegionType
    lines: List[TextLine] = field(default_factory=list)

    def get_text(self, separator: str = "\n") -> str:
        return separator.join(str(line) for line in self.lines)

@dataclass
class TextLine:
    coords: str
    id: str
    words: List[Word] = field(default_factory=list)

    def __str__(self) -> str:
        return " ".join(word.text for word in self.words)

    def set_linescore(
            self,
            error_type: ErrorType,
            score: float,
    ) -> None:
        for w in self.words:
            w.word_error_score = score
            w.error_type = error_type

@dataclass
class Word:
    text: str
    word_error_score: Optional[float] = None  # 0.0 – 1.0, lower is better
    error_type: ErrorType = ErrorType.NONE

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

def _enum_from_string(enum_cls, value: str, default):
    key = value.replace("-", "_").replace(" ", "_").upper()
    return enum_cls.__members__.get(key, default)


def _split_into_words(text: str) -> List[str]:
    return [token for token in text.strip().split() if token]

def load_pages(folder: Path, image_path: Optional[Path] = Path(__file__).resolve().parents[2] / "data" / "d2_0001-0100_without_marginalia") -> List[Page]:
    xml_files: List[Path] = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".xml"
    ]
    return [parse_page_from_annotated_xml(p, image_path) for p in xml_files]


def save_pages(pages: List[Page], output_folder: Path) -> List[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for page in pages:                        # preserve incoming order
        out_path = parse_annotated_xml_from_page(page, output_folder)
        written.append(out_path)
    return written




def parse_annotated_xml_from_page(page: Page, output_path: Path) -> Path:
    if output_path.is_dir() or output_path.suffix == "":
        output_file = output_path / f"{page.name}.xml"
    else:
        output_file = output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # root <Page> -------------------------------------------------------------
    page_el = ET.Element(
        "Page",
        {

            "imageFilename": page.image_path.name
                             if page.image_path else page.name,
            "imageHeight": str(page.imageHeight),
            "imageWidth":  str(page.imageWidth),
            "orientation": "0.0",
            "overwritePageError": str(page.overwrite_page_error),
        },
    )

    # regions → lines → words -------------------------------------------------
    for region in page.text_regions:
        region_el = ET.SubElement(
            page_el, "TextRegion",
            {"id": region.id, "type": region.type.name.lower()}
        )
        ET.SubElement(region_el, "Coords", {"points": region.coords})

        for line in region.lines:
            line_el = ET.SubElement(
                region_el, "TextLine",
                {"id": line.id, "text": str(line)}   # ← line text attribute
            )
            ET.SubElement(line_el, "Coords", {"points": line.coords})

            for idx, word in enumerate(line.words, 1):
                attrs = {
                    "id":   f"{line.id}-w_{idx:04d}",
                    "text": word.text,                 # ← word text attribute
                }
                if word.word_error_score is not None:
                    attrs["error_score"] = f"{word.word_error_score:.3f}"
                if word.error_type is not ErrorType.NONE:
                    attrs["error_type"] = word.error_type.name.lower()

                ET.SubElement(line_el, "Word", attrs)

    # pretty-print (ElementTree ≥3.9) ----------------------------------------
    tree = ET.ElementTree(page_el)
    if hasattr(ET, "indent"):                # Python 3.9+
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
    else:                                     # fallback for 3.8 and earlier
        import xml.dom.minidom as minidom
        xml_str = ET.tostring(page_el, encoding="utf-8")
        pretty_xml = (
            minidom.parseString(xml_str)
            .toprettyxml(indent="  ", encoding="utf-8")
        )
        output_file.write_bytes(pretty_xml)

    return output_file


def parse_page_from_annotated_xml(xml_path: Path, image_path : Path) -> Page:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page = Page(xml_path=xml_path)

    page.imageHeight = int(root.attrib.get("imageHeight", 0))
    page.imageWidth = int(root.attrib.get("imageWidth", 0))
    page.image_path = (
        (image_path / root.attrib["imageFilename"]).resolve()
        if "imageFilename" in root.attrib
        else None
    )
    page.overwrite_page_error = float(root.attrib.get("overwritePageError", -1))

    for region_el in root.findall("TextRegion"):
        coords_el = region_el.find("Coords")
        region = TextRegion(
            coords=coords_el.attrib.get("points", "") if coords_el is not None else "",
            id=region_el.attrib["id"],
            type=_region_type_from_xml(region_el.attrib.get("type", "")),
        )

        for line_el in region_el.findall("TextLine"):
            line_coords_el = line_el.find("Coords")
            line = TextLine(
                coords=line_coords_el.attrib.get("points", "") if line_coords_el is not None else "",
                id=line_el.attrib["id"],
            )

            for word_el in line_el.findall("Word"):
                word = Word(
                    text=word_el.attrib.get("text", ""),
                    word_error_score=(
                        float(word_el.attrib["error_score"])
                        if "error_score" in word_el.attrib
                        else None
                    ),
                    error_type=_error_type_from_xml(word_el.attrib.get("error_type", "")),
                )
                line.words.append(word)

            region.lines.append(line)

        page.text_regions.append(region)

    return page

def parse_raw_page_from_xml(xml_path: Path) -> Page:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page_el = next((el for el in root.iter() if _local(el.tag) == "Page"), None)
    if page_el is None:
        raise ValueError("No <Page> element found in XML")

    page = Page(xml_path=xml_path)

    img_filename = page_el.attrib.get("imageFilename")
    page.imageHeight = page_el.attrib.get("imageHeight")
    page.imageWidth = page_el.attrib.get("imageWidth")
    if img_filename:
        # If the filename ends with .png but not .nrm.png, convert it to .nrm.png
        if img_filename.lower().endswith(".png") and not img_filename.lower().endswith(".bin.png"):
            stem = img_filename[:-4]  # removes ".png"
            img_filename = f"{stem}.bin.png"
        page.image_path = (xml_path.parent / img_filename).resolve()
    else:
        raise ValueError(f"No imageFilename attribute found in XML file: {xml_path}")

    for region_el in (el for el in page_el if _local(el.tag) == "TextRegion"):
        r_id = region_el.attrib.get("id", "")
        r_type_str = region_el.attrib.get("type", "paragraph")
        r_type = _enum_from_string(RegionType, r_type_str, RegionType.OTHER)

        coords_el = next((c for c in region_el if _local(c.tag) == "Coords"), None)
        r_coords = coords_el.attrib.get("points", "") if coords_el is not None else ""

        region = TextRegion(coords=r_coords, id=r_id, type=r_type)

        # ---- lines ---------------------------------------------------------
        for line_el in (el for el in region_el if _local(el.tag) == "TextLine"):
            l_id = line_el.attrib.get("id", "")
            l_coords_el = next((c for c in line_el if _local(c.tag) == "Coords"), None)
            l_coords = l_coords_el.attrib.get("points", "") if l_coords_el is not None else ""

            unicode_text = None
            for te in (child for child in line_el if _local(child.tag) == "TextEquiv"):
                uni = next((u for u in te if _local(u.tag) == "Unicode"), None)
                if uni is not None and uni.text:
                    unicode_text = uni.text
                    break

            words_text: List[str] = []

            if unicode_text:
                words_text = _split_into_words(unicode_text)
            else:
                for w_el in (el for el in line_el if _local(el.tag) == "Word"):
                    for te in (child for child in w_el if _local(child.tag) == "TextEquiv"):
                        uni = next((u for u in te if _local(u.tag) == "Unicode"), None)
                        if uni is not None and uni.text:
                            words_text.append(uni.text.strip())

            line = TextLine(coords=l_coords, id=l_id, words=[Word(text=w) for w in words_text])
            region.lines.append(line)

        page.text_regions.append(region)
    return page

# ---------------------- private  methods -----------------------------
def _coords_to_pts(coords: str) -> List[Tuple[int, int]]:
    return [
        (int(x), int(y))
        for x, y in (p.split(",") for p in coords.strip().split() if "," in p)
    ]


def _fill_single_layer(
    base: Image.Image,
    polys: List[List[Tuple[int, int]]],
    colour: str,
    opacity: int = 80,
) -> None:
    from PIL import Image, ImageDraw, ImageColor

    if not polys:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    rgba = ImageColor.getrgb(colour) + (opacity,)
    for poly in polys:
        draw.polygon(poly, fill=rgba)
    base.alpha_composite(layer)

def _region_type_from_xml(value: str) -> RegionType:
    """
    Convert the lower-case string stored in the XML back to a RegionType enum.
    Unknown values fall back to RegionType.OTHER (rather than raising).
    """
    lookup: Dict[str, RegionType] = {t.name.lower(): t for t in RegionType}
    return lookup.get(value.lower(), RegionType.OTHER)


def _error_type_from_xml(value: str) -> ErrorType:
    """
    Convert the lower-case string stored in the XML back to an ErrorType enum.
    Missing / “none” → ErrorType.NONE.
    """
    if not value:
        return ErrorType.NONE
    lookup: Dict[str, ErrorType] = {t.name.lower(): t for t in ErrorType}
    return lookup.get(value.lower(), ErrorType.NONE)


def visualize_text_regions(page: Page) -> Image.Image:
    """
    Colour-codes *TextRegion* polygons using the Larex palette.
    """
    from PIL import Image, ImageDraw, ImageFont

    if page.image_path is None or not page.image_path.exists():
        raise FileNotFoundError("`image_path` not set or file does not exist")

    img = Image.open(page.image_path).convert("RGBA")

    REGION_COLORS: dict[RegionType, Tuple[str, str]] = {
        RegionType.PAGE_NUMBER: ("Page Number", "lightskyblue"),
        RegionType.PARAGRAPH: ("Paragraph", "red"),
        RegionType.MARGINALIA: ("Marginalia", "yellow"),
        RegionType.HEADER: ("Header", "brown"),
        RegionType.HEADING: ("Header", "brown"),
        RegionType.FOOTNOTE: ("Footnote", "orange"),
        RegionType.OTHER: ("Other", "darkgreen"),
    }

    bucket: dict[RegionType, List[List[Tuple[int, int]]]] = {rt: [] for rt in REGION_COLORS}
    for region in page.text_regions:
        if not region.coords:
            continue
        pts = _coords_to_pts(region.coords)
        if pts:
            key = RegionType[region.type.name]  # convert to the canonical class
            bucket[key].append(pts)

    for r_type, polys in bucket.items():
        _, col = REGION_COLORS[r_type]
        _fill_single_layer(img, polys, col, opacity=80)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    bw, bh, gap, mar = 30, 20, 6, 15
    legend = [(lbl, col) for lbl, col in REGION_COLORS.values()]
    panel_w = bw + 8 + max(font.getlength(lbl) if hasattr(font, "getlength") else font.getsize(lbl)[0] for lbl, _ in legend)
    panel_h = len(legend) * bh + (len(legend) - 1) * gap
    xs0, ys0 = mar, img.height - panel_h - mar
    draw.rectangle([xs0 - 6, ys0 - 6, xs0 + panel_w + 6, ys0 + panel_h + 6], fill=(0, 0, 0, 120))

    ys = ys0
    for lbl, col in legend:
        draw.rectangle([xs0, ys, xs0 + bw, ys + bh], fill=col, outline="black")
        th = font.getsize(lbl)[1] if hasattr(font, "getsize") else bh
        draw.text((xs0 + bw + 8, ys + (bh - th) // 2), lbl, "white", font)
        ys += bh + gap

    return img


# -------- utility for text dimensions (safe across Pillow versions) ----
def _text_dims(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """
    Return (width, height) of *text* for *font*.

    Uses `textbbox` when available (Pillow ≥ 8.0) and falls back to
    `font.getbbox` / `font.getsize` otherwise.
    """
    if hasattr(draw, "textbbox"):                    # preferred
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    if hasattr(font, "getbbox"):                     # Pillow 9.x
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    return font.getsize(text)                        # legacy

def visualize_error_regions(page: Page) -> Image.Image:
    """
    Overlay each *TextLine* polygon with a colour reflecting the average
    `word_error_score` of its words.
    """
    from PIL import Image, ImageDraw, ImageFont

    def _score_to_rgba(score: float, alpha: int = 120) -> Tuple[int, int, int, int]:
        score = max(0.0, min(1.0, score))
        if score <= 0.5:                   # green → yellow
            t = score / 0.5
            r, g = int(255 * t), 255
        else:                              # yellow → red
            t = (score - 0.5) / 0.5
            r, g = 255, int(255 * (1 - t))
        return r, g, 0, alpha

    if page.image_path is None or not page.image_path.exists():
        raise FileNotFoundError("`image_path` not set or file does not exist")

    base    = Image.open(page.image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ovl = ImageDraw.Draw(overlay, "RGBA")

    # --- draw polygons ----------------------------------------------------
    for region in page.text_regions:
        for line in region.lines:
            if not line.coords:
                continue
            pts = _coords_to_pts(line.coords)
            if not pts:
                continue
            scores = [w.word_error_score for w in line.words if w.word_error_score is not None]
            if scores:  # list is non-empty
                avg = sum(scores) / len(scores)
            else:  # no scores → treat as 100 % error
                avg = 1.0

            draw_ovl.polygon(pts, fill=_score_to_rgba(avg))

    base.alpha_composite(overlay)

    # --- legend & colour-bar ------------------------------------------------
    draw = ImageDraw.Draw(base)

    # choose generous sizes so the whole legend is easy to read
    try:
        font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)   # numbers
        font_tick = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)   # 0, 0.5, 1
    except OSError:
        font_info = font_tick = ImageFont.load_default()

    # 1)  compute global scores ---------------------------------------------
    err, tot = page.get_amount_of_error_vs_total()
    rel_err  = page.calculate_total_relative_error_score()          # 0‥1

    # 2)  geometry of the legend block (bottom-left) -------------------------
    mar       = 25                        # margin from canvas edge
    bar_w     = 400                       # width of colour bar
    bar_h     = 40                        # height of colour bar
    line_gap  = 6                         # vertical gap between text lines

    # top-left corner of the legend block
    x0        = mar
    y0        = base.height - (bar_h + 2*font_info.size + 3*line_gap) - mar

    # 3)  paint the numerical info above the bar -----------------------------
    txt1 = f"Error / Total: {err} / {tot}"
    txt2 = f"ErrorScore:    {rel_err*100:5.1f} %"

    draw.text((x0, y0),              txt1, fill="black", font=font_info)
    draw.text((x0, y0 + font_info.size + line_gap),
              txt2, fill="black", font=font_info)

    # update y-pos for the colour bar
    y_bar = y0 + 2*font_info.size + 2*line_gap

    # 4)  draw the colour bar itself ----------------------------------------
    for i in range(bar_w):
        s = i / (bar_w - 1)
        draw.line(
            [(x0 + i, y_bar), (x0 + i, y_bar + bar_h)],
            fill=_score_to_rgba(s, 255)
        )

    # 5)  tick labels centred under the bar ----------------------------------
    for lbl, xpos in (
        ("0",   x0),
        ("0.5", x0 + bar_w // 2),
        ("1.0", x0 + bar_w),
    ):
        w, h = _text_dims(draw, lbl, font_tick)
        draw.text((xpos - w // 2, y_bar + bar_h + 6),
                  lbl, "black", font=font_tick)

    return base

def visualize_text_errors(page: Page) -> Image.Image:
    from PIL import Image, ImageDraw, ImageFont

    def _score_to_rgba(score: float, alpha: int = 255) -> tuple[int, int, int, int]:
        """
        Map `score` ∈ [0, 1] → RGBA:
            0.0 → green, 0.5 → yellow, 1.0 → red
        """
        score = max(0.0, min(1.0, score))
        if score <= 0.5:                      # green → yellow
            t = score / 0.5
            r, g = int(255 * t), 255
        else:                                 # yellow → red
            t = (score - 0.5) / 0.5
            r, g = 255, int(255 * (1 - t))
        return r, g, 0, alpha

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=40)
    except OSError:
        font = ImageFont.load_default()

    margin     = 40                       # space around the text block
    line_gap   = 12                       # vertical gap between lines
    space_w, _ = font.getbbox(" ")[2:]    # width of a single space (for word gaps)

    lines: list[list[Word]] = []
    max_w_px = 0
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    for region in page.text_regions:
        for line in region.lines:
            if not line.words:
                continue
            lines.append(line.words)

            # width of this line in pixels
            w_sum = 0
            for idx, word in enumerate(line.words):
                bbox = dummy_draw.textbbox((0, 0), word.text, font=font)
                w_sum += bbox[2] - bbox[0]          # word width
                if idx < len(line.words) - 1:       # space after word (except last)
                    w_sum += space_w

            max_w_px = max(max_w_px, w_sum)

    if not lines:
        # page has no text at all → return an empty white canvas
        empty = Image.new("RGBA", (800, 400), "white")
        return empty

    line_height = font.getbbox("Hg")[3] - font.getbbox("Hg")[1]   # ascent+descent
    canvas_w = max_w_px + 2 * margin
    canvas_h = len(lines) * (line_height + line_gap) - line_gap + 2 * margin

    # ------------------------------------------------------------------ draw pass 2: paint
    canvas = Image.new("RGBA", (canvas_w, canvas_h), "white")
    draw   = ImageDraw.Draw(canvas)

    y_cursor = margin
    for words in lines:
        x_cursor = margin
        for idx, word in enumerate(words):
            score = (
                word.word_error_score
                if word.word_error_score is not None
                else 1.0                        # missing score → treat as worst (red)
            )
            draw.text(
                (x_cursor, y_cursor),
                word.text,
                font=font,
                fill=_score_to_rgba(score),
            )

            # advance cursor
            word_w = draw.textbbox((x_cursor, y_cursor), word.text, font=font)[2] - x_cursor
            x_cursor += word_w
            if idx < len(words) - 1:
                x_cursor += space_w

        y_cursor += line_height + line_gap

    return canvas

def reduce_to_common_page_by_name(pages1 : List[Page], pages2 : List[Page]):
    # Get the set of page names from each list
    names1 = {page.name for page in pages1}
    names2 = {page.name for page in pages2}

    # Determine the intersection (common names)
    common_names = names1 & names2

    # Filter each list in-place
    pages1[:] = [page for page in pages1 if page.name in common_names]
    pages2[:] = [page for page in pages2 if page.name in common_names]