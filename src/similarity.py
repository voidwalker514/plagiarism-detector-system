"""
similarity.py
-------------
Similarity Score Calculation Module.

Combines results from multiple string-matching algorithms to produce
a unified plagiarism percentage and categorized verdict.

DSA concepts used:
    - Set intersection / union (Jaccard similarity)
    - HashMap / Counter (word frequency mapping)
    - Sorted list (for ranking matched sentences)
"""

from collections import Counter


# ---------------------------------------------------------------------------
# 1. Sentence-level Jaccard Similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Compute the Jaccard Similarity Coefficient between two sets.

    Formula:   |A ∩ B| / |A ∪ B|

    DSA concept: Set operations — intersection and union.

    Args:
        set_a: First set (e.g., token set of original document).
        set_b: Second set (e.g., token set of submitted document).

    Returns:
        Float in range [0.0, 1.0] representing similarity.
        1.0 = identical, 0.0 = completely different.
    """
    if not set_a and not set_b:
        return 1.0  # both empty → identical

    intersection = set_a & set_b   # common elements
    union        = set_a | set_b   # all elements

    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# 2. N-gram Overlap Similarity
# ---------------------------------------------------------------------------

def ngram_overlap_similarity(ngrams_orig: list, ngrams_sub: list) -> float:
    """
    Compute similarity as: matched n-grams / total n-grams in submitted doc.

    This is the primary similarity metric used for plagiarism percentage.

    Args:
        ngrams_orig: List of n-gram strings from original document.
        ngrams_sub:  List of n-gram strings from submitted document.

    Returns:
        Float in range [0.0, 1.0].
    """
    orig_set = {ng.lower() for ng in ngrams_orig}  # O(k) hash set build

    if not ngrams_sub:
        return 0.0

    matched = sum(1 for ng in ngrams_sub if ng.lower() in orig_set)
    return matched / len(ngrams_sub)


# ---------------------------------------------------------------------------
# 3. Word Frequency (TF-style) Similarity
# ---------------------------------------------------------------------------

def word_frequency_similarity(tokens_orig: list, tokens_sub: list) -> float:
    """
    Compute cosine-style similarity using word frequency counts.

    Steps:
        1. Build frequency Counter (HashMap) for each document.
        2. Find common vocabulary.
        3. Compute dot product of frequency vectors.
        4. Normalize by magnitudes.

    DSA concept: HashMap (Counter) for frequency tracking.

    Args:
        tokens_orig: List of word tokens from original document.
        tokens_sub:  List of word tokens from submitted document.

    Returns:
        Float in range [0.0, 1.0].
    """
    freq_orig = Counter(tokens_orig)   # word → count (HashMap)
    freq_sub  = Counter(tokens_sub)

    # Common words (vocabulary intersection)
    common = set(freq_orig.keys()) & set(freq_sub.keys())

    if not common:
        return 0.0

    # Dot product
    dot_product = sum(freq_orig[w] * freq_sub[w] for w in common)

    # Magnitudes
    mag_orig = sum(v ** 2 for v in freq_orig.values()) ** 0.5
    mag_sub  = sum(v ** 2 for v in freq_sub.values())  ** 0.5

    if mag_orig == 0 or mag_sub == 0:
        return 0.0

    return dot_product / (mag_orig * mag_sub)


# ---------------------------------------------------------------------------
# 4. Sentence-level Match Rate
# ---------------------------------------------------------------------------

def sentence_match_rate(sentences_orig: list, sentences_sub: list) -> float:
    """
    Compute what fraction of submitted sentences exactly appear in the original.

    Args:
        sentences_orig: List of sentences from original document.
        sentences_sub:  List of sentences from submitted document.

    Returns:
        Float in range [0.0, 1.0].
    """
    orig_set = {s.lower().strip() for s in sentences_orig}

    if not sentences_sub:
        return 0.0

    matched = sum(1 for s in sentences_sub if s.lower().strip() in orig_set)
    return matched / len(sentences_sub)


# ---------------------------------------------------------------------------
# 5. Aggregate Plagiarism Score
# ---------------------------------------------------------------------------

def calculate_plagiarism_score(
    ngrams_orig:     list,
    ngrams_sub:      list,
    tokens_orig:     list,
    tokens_sub:      list,
    sentences_orig:  list,
    sentences_sub:   list,
    weights: dict = None
) -> dict:
    """
    Compute a weighted aggregate plagiarism score from multiple metrics.

    Weights (default):
        - n-gram overlap:           60%
        - sentence exact match:     25%
        - word frequency cosine:    15%

    Args:
        ngrams_orig / ngrams_sub:       N-gram lists for both documents.
        tokens_orig / tokens_sub:       Token lists for both documents.
        sentences_orig / sentences_sub: Sentence lists for both documents.
        weights: Optional dict to override default weights.

    Returns:
        Dict containing all individual scores and the final weighted score.
    """
    if weights is None:
        weights = {
            "ngram_overlap":   0.60,
            "sentence_match":  0.25,
            "word_frequency":  0.15,
        }

    # Individual metric scores
    ng_score   = ngram_overlap_similarity(ngrams_orig, ngrams_sub)
    sent_score = sentence_match_rate(sentences_orig, sentences_sub)
    wf_score   = word_frequency_similarity(tokens_orig, tokens_sub)
    jac_score  = jaccard_similarity(set(tokens_orig), set(tokens_sub))

    # Weighted aggregate
    weighted = (
        weights["ngram_overlap"]  * ng_score   +
        weights["sentence_match"] * sent_score +
        weights["word_frequency"] * wf_score
    )

    plagiarism_pct = round(weighted * 100, 2)

    return {
        "ngram_overlap_score":   round(ng_score   * 100, 2),
        "sentence_match_score":  round(sent_score * 100, 2),
        "word_frequency_score":  round(wf_score   * 100, 2),
        "jaccard_score":         round(jac_score  * 100, 2),
        "plagiarism_percentage": plagiarism_pct,
        "verdict":               get_verdict(plagiarism_pct),
    }


# ---------------------------------------------------------------------------
# 6. Verdict / Classification
# ---------------------------------------------------------------------------

def get_verdict(percentage: float) -> str:
    """
    Classify the plagiarism percentage into a human-readable verdict.

    Thresholds (industry-standard approximation):
        0-10%   → Original / Safe
        11-30%  → Minor Similarity
        31-50%  → Moderate Plagiarism
        51-75%  → High Plagiarism
        76-100% → Severe Plagiarism

    Args:
        percentage: Plagiarism percentage (0-100).

    Returns:
        Verdict string.
    """
    if percentage <= 10:
        return "✅ ORIGINAL  — Safe (0-10%)"
    elif percentage <= 30:
        return "🟡 MINOR SIMILARITY  — Review Recommended (11-30%)"
    elif percentage <= 50:
        return "🟠 MODERATE PLAGIARISM  — Action Required (31-50%)"
    elif percentage <= 75:
        return "🔴 HIGH PLAGIARISM  — Serious Concern (51-75%)"
    else:
        return "🚨 SEVERE PLAGIARISM  — Direct Copy Detected (76-100%)"
