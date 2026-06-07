"""
preprocessor.py
---------------
Text Preprocessing Module for Plagiarism Detector.

Responsibilities:
  - Read text files from disk
  - Clean / normalize raw text
  - Split text into sentences and word-tokens
  - Build n-gram phrase windows (for phrase-level matching)

DSA concepts used:
  - List (dynamic array) for storing tokens / sentences
  - String slicing & sliding window for n-gram generation
"""

import re
import os


# ---------------------------------------------------------------------------
# 1. File Reading
# ---------------------------------------------------------------------------

def read_file(filepath: str) -> str:
    """
    Read a plain-text file and return its contents as a string.

    Args:
        filepath: Absolute or relative path to the text file.

    Returns:
        Raw text content as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        content = fh.read()

    return content


# ---------------------------------------------------------------------------
# 2. Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize raw text for consistent comparison.

    Steps:
      1. Convert to lowercase so comparisons are case-insensitive.
      2. Remove punctuation (keep alphanumeric & whitespace).
      3. Collapse multiple spaces / newlines into a single space.
      4. Strip leading/trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized text string.
    """
    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove punctuation (keep letters, digits, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Step 3 – collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Step 4 – strip
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# 3. Sentence Splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list:
    """
    Split raw (un-cleaned) text into individual sentences.

    We split on sentence-ending punctuation followed by whitespace.
    Empty strings are filtered out.

    Args:
        text: Raw text (before cleaning) so punctuation is still present.

    Returns:
        List of sentence strings (stripped).
    """
    # Split on '.', '!', '?' followed by whitespace or end-of-string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Filter blanks
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


# ---------------------------------------------------------------------------
# 4. Word Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Split a (cleaned) text string into individual word tokens.

    Args:
        text: Cleaned text string.

    Returns:
        List of word tokens (all lowercase, no punctuation).
    """
    return text.split()


# ---------------------------------------------------------------------------
# 5. N-Gram Generation  (Sliding Window)
# ---------------------------------------------------------------------------

def generate_ngrams(tokens: list, n: int = 5) -> list:
    """
    Generate n-gram phrases from a list of tokens using a sliding window.

    A n-gram is a contiguous sequence of n words.  We use these as the
    "phrases" that string-matching algorithms search for.

    Example (n=3):
        tokens = ["the", "cat", "sat", "on", "mat"]
        ngrams = ["the cat sat", "cat sat on", "sat on mat"]

    DSA concept: Sliding Window — O(n) time, O(k) extra space per window.

    Args:
        tokens: List of word tokens.
        n:      Window size (number of words per phrase).

    Returns:
        List of phrase strings.
    """
    if len(tokens) < n:
        # If document is shorter than n, treat whole text as one phrase
        return [" ".join(tokens)]

    ngrams = []
    # Slide a window of size n across the token list
    for i in range(len(tokens) - n + 1):
        phrase = " ".join(tokens[i : i + n])
        ngrams.append(phrase)

    return ngrams


# ---------------------------------------------------------------------------
# 6. High-level Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess(filepath: str, ngram_size: int = 5) -> dict:
    """
    Full preprocessing pipeline for a single document.

    Combines reading, cleaning, sentence splitting, tokenization,
    and n-gram generation into one convenient call.

    Args:
        filepath:   Path to the text file.
        ngram_size: Size of n-gram window (default 5 words).

    Returns:
        Dictionary with keys:
            'raw'       – original file text
            'cleaned'   – normalized text
            'sentences' – list of raw sentences
            'tokens'    – list of word tokens
            'ngrams'    – list of n-gram phrase strings
    """
    raw_text = read_file(filepath)
    cleaned = clean_text(raw_text)
    sentences = split_sentences(raw_text)
    tokens = tokenize(cleaned)
    ngrams = generate_ngrams(tokens, n=ngram_size)

    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "ngrams": ngrams,
    }
