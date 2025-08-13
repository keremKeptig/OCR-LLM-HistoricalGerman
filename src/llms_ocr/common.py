import torch
import math
import re
import unicodedata

# ------------------------------Chunking Analysis -------------------------------


def is_page_number(paragraph: str) -> bool:
    """Checks if the paragraph is a page number.
    Args:
        paragraph (str): The input paragraph to check.
    Returns:
        bool: True if the paragraph is a page number, False otherwise.
    """
    pattern = re.compile(r"^(?:[—\-]\s*)?\d+(?:\s*[—\-])?$")
    match = bool(pattern.fullmatch(paragraph.strip()))
    return match


def merge_split_words(text):
    """Merges split words in the text.
    Args:
        text (str): The input text from which to merge split words.
    Returns:
        str: The text with split words merged.
    """
    return re.sub(r"⸗\n*", "", text)


def remove_newlines(text):
    """Removes excessive newlines from the text.
    Args:
        text (str): The input text from which to remove newlines.
    Returns:
        str: The text with excessive newlines removed, replaced by a single space.
    """
    return re.sub(r"\n+", " ", text)


def perplexity(text: str, model, tokenizer, max_length=512) -> float:
    """Calculates the perplexity of a given text using a language model.
    Args:
        text (str): The input text for which to calculate perplexity.
        model: The language model to use for perplexity calculation.
        tokenizer: The tokenizer corresponding to the language model.
        max_length (int): The maximum length of the input text to consider.
    Returns:
        float: The perplexity of the input text.
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        return math.exp(loss.item())


def get_raw_paragraphs(xml_doc=None):
    """Extracts raw paragraphs from an XML document.
    Args:
        xml_doc: An XML document object containing text regions and lines.
    Returns:
        dict: A dictionary where keys are paragraph IDs and values are the concatenated text of lines
              within each paragraph.
    """
    raw_paragraphs = {}
    for paragraph in xml_doc.text_regions:
        text = "\n".join(line.text for line in paragraph.lines if line.text is not None)
        raw_paragraphs[paragraph.id] = text
    return raw_paragraphs


def chunk_text_by_sentence(paragraph_text, nlp_model, window_size=2, overlap_size=1):
    """Splits the paragraph text into chunks of sentences with a specified overlap.
    Args:
        paragraph_text (str): The input paragraph text to split.
        nlp_model: A spaCy NLP model for sentence segmentation.
        window_size (int): The number of sentences in each chunk.
        overlap_size (int): The number of sentences to overlap between chunks.
    Returns:
        list: A list of strings, each containing a chunk of sentences.
    """
    doc = nlp_model(paragraph_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    step = window_size - overlap_size

    if step <= 0:
        # Avoid infinite loop if overlap is too large or equal to window_size
        print("Warning: overlap_size is too large. Setting step to 1.")
        step = 1

    for i in range(0, len(sentences), step):
        # Define the end of the current window
        end_index = min(i + window_size, len(sentences))

        # Join the sentences within the current window to form a single chunk string
        current_chunk_sentences = sentences[i:end_index]
        chunks.append(" ".join(current_chunk_sentences))

        # If the window reached the end of the sentences, stop
        if end_index == len(sentences):
            break

    return chunks


def chunk_text_by_word(text, chunk_size=15, overlap_size=5):
    """Splits the text into chunks of words with a specified overlap.

    Args:
        text (str): The input text to split.
        chunk_size (int): The size of each chunk (in words).
        overlap_size (int): The size of the overlap between chunks (in words).

    Returns:
        list: A list of dictionaries, each containing the chunk text and its word IDs.
    """

    words = text.split()
    processed_chunks_data = []  # This will store our list of dictionaries
    step = chunk_size - overlap_size

    if not words:
        print("No words found in the text. Returning empty list.")
        return []

    if step <= 0:
        print(
            f"Overlap_size ({overlap_size}) is too large or equal to chunk_size ({chunk_size}). "
            f"Setting step to 1 to ensure progress."
        )
        step = 1

    chunk_counter = 0

    for i in range(0, len(words), step):
        end_word_index_in_words_list = min(i + chunk_size, len(words))
        chunk_words_list = words[i:end_word_index_in_words_list]
        chunk_text = " ".join(chunk_words_list)

        start_word_id = i
        end_word_id = start_word_id + len(chunk_words_list) - 1

        processed_chunks_data.append(
            {
                "chunk_id": chunk_counter,
                "text": chunk_text,
                "start_word_id": start_word_id,
                "end_word_id": end_word_id,
            }
        )

        chunk_counter += 1

        if end_word_index_in_words_list == len(words):
            break

    return processed_chunks_data


def get_word_to_line_mapping(xml_doc=None):
    """
    Simple function to map every word to its line ID.

    Returns:
        dict: {paragraph_id: [line_id, line_id, ...]}
              where each line_id corresponds to a word position
    """
    word_line_mappings = {}

    for paragraph in xml_doc.text_regions:
        word_line_ids = []

        for line in paragraph.lines:
            if line.text is not None:
                words_in_line = line.text.split()
                # Add the line ID for each word in this line
                word_line_ids.extend([line.id] * len(words_in_line))

        word_line_mappings[paragraph.id] = word_line_ids

    return word_line_mappings


# ----------------------------------Rule based German Sentence Splitter--------------------------------------

# Splits German text into sentences while preserving abbreviations and ellipses. Includes fixes for historical chars, and hyphen breaks.

# German Abbreviation List
GERMAN_ABBREVIATIONS = {
    # Common
    "z.B.",
    "u.a.",
    "u.U.",
    "d.h.",
    "ggf.",
    "bzw.",
    "etc.",
    "sog.",
    "ca.",
    "Nr.",
    "Dr.",
    "Prof.",
    "bspw.",
    "vgl.",
    "Fig.",
    "Abb.",
    "Hr.",
    "Fr.",
    "bspw.",
    "evtl.",
    "ggf.",
    "bsp.",
    "i.d.R.",
    "Jan.",
    "Feb.",
    "Mrz.",
    "Apr.",
    "Mai.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Okt.",
    "Nov.",
    "Dez.",
    "Jh.",
    "St.",
    "Abs.",
    "S.",
    "Tab.",
    "Bd.",
    "Hrsg.",
    "Kap.",
    "Lit.",
    "s.",
    "Anm.",
    "bspw.",
    "dgl.",
    "u.Ä.",
    "usw.",
    "z.T.",
    "z.Z.",
    "zzgl.",
    # Add more as needed for your documents
}

# Helpers for abbreviation protection and splitting


def replace_historical_chars(text):
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ſ": "s",
        "ʒ": "z",  # LATIN SMALL LETTER EZH
        "Ʒ": "Z",  # LATIN CAPITAL LETTER EZH
        "ƶ": "z",  # LATIN SMALL LETTER EZH WITH CURL
    }
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    return text


def fix_linebreak_hyphens(text):
    pattern = re.compile(r"(\w+)[-\u2E17]\s+([a-zäöüß])", re.UNICODE)
    while True:
        new_text, count = pattern.subn(r"\1\2", text)
        if count == 0:
            break
        text = new_text
    return text


# Rule-based sentence splitting
def rule_based_split(
    text, allow_lowercase_start=True, abbreviations=GERMAN_ABBREVIATIONS
):
    text = replace_historical_chars(text)
    text = fix_linebreak_hyphens(text)

    # Abbreviation protection as before
    abbrev_core = [re.escape(abbr[:-1]) for abbr in abbreviations]
    abbrev_pattern = re.compile(
        r"\b(" + "|".join(abbrev_core) + r")\."  # Match abbr root + .
        r"(\s*\d+(\.\d+)*)?"  # Optional numbers/dots
        r"\.",  # Final .
        re.IGNORECASE,
    )

    def replace_dots(m):
        return m.group(0).replace(".", "<DOT>")

    text = abbrev_pattern.sub(replace_dots, text)

    # Protect ellipsis
    text = text.replace("...", "<ELLIPSIS>")

    # Sentence-ending pattern (fixed-width lookbehind)
    # Match: . or ! or ? followed by optional " ' ) ] etc., whitespace, and then capital/lowercase
    next_start = "A-ZÄÖÜ"
    if allow_lowercase_start:
        next_start += "a-zäöüß"
    # Pattern: ([.!?][\"')\]\u201d\u2019\u00bb\u203a]?)(\s+)(?=[A-Za-zÄÖÜäöüß])
    sentence_boundary = re.compile(
        r"([.!?][\"\'\)\]\u201d\u2019\u00bb\u203a]?)(\s+)(?=[" + next_start + "])"
    )

    # Insert split marker
    text = sentence_boundary.sub(r"\1<SPLIT>", text)

    # Split on marker
    sentences = text.split("<SPLIT>")

    # Restore protected
    sentences = [
        s.replace("<DOT>", ".").replace("<ELLIPSIS>", "...") for s in sentences
    ]
    return [s.strip() for s in sentences if s.strip()]
