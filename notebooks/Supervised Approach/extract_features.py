import numpy as np
from src.evaluator.page_parser import Page


def extract_filtered_features_from_page(page: Page, embedder) -> np.ndarray:
    lines, words, chars_per_line = [], [], []
    lines_per_region, words_per_region = [], []
    empty_regions = 0
    total_regions = len(page.text_regions)
    empty_lines = 0
    total_lines = 0
    for region in page.text_regions:
        region_lines = getattr(region, 'lines', [])
        if not region_lines:
            empty_regions += 1
        lines_per_region.append(len(region_lines))
        region_words = []
        for line in region_lines:
            line_words = [word.text for word in getattr(line, 'words', [])]
            # Count empty line if no words or only whitespace
            if not line_words or not "".join(line_words).strip():
                empty_lines += 1
            total_lines += 1
            chars_per_line.append(len(" ".join(line_words)))
            region_words.extend(line_words)
            lines.append(line_words)
            words.extend(line_words)
        words_per_region.append(len(region_words))
    n_lines, n_words = len(lines), len(words)
    handcrafted = np.array([
        len(page.text_regions), n_lines, n_words,
        sum(len(w) for w in words) / n_words if n_words else 0,
        np.mean(lines_per_region) if lines_per_region else 0,
        np.var(lines_per_region) if lines_per_region else 0,
        np.mean(words_per_region) if words_per_region else 0,
        np.var(words_per_region) if words_per_region else 0,
        max(chars_per_line) if chars_per_line else 0,
        np.mean(chars_per_line) if chars_per_line else 0,
        np.var(chars_per_line) if chars_per_line else 0,
        sum(w.isnumeric() for w in words) / n_words if n_words else 0,
        sum(all(c in ".,;:!?-" for c in w) for w in words) / n_words if n_words else 0,
        max((len(w) for w in words), default=0),
        # --- min_word_len is removed ---
        # --- New features: empty regions and lines ---
        empty_regions,                                  # Number of empty regions (paragraphs)
        empty_regions / total_regions if total_regions else 0,  # Proportion of empty regions
        empty_lines,                                    # Number of empty lines
        empty_lines / total_lines if total_lines else 0,        # Proportion of empty lines
    ])
    embedding = embedder.encode([" ".join(words)])[0]
    return np.concatenate([embedding, handcrafted])