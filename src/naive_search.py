"""
naive_search.py
---------------
Naive (Brute-Force) String Matching Algorithm.

Algorithm Description:
    Slide a window of length |pattern| across the text character by character.
    At every position, compare each character of the pattern with the
    corresponding character in the text.  Report a match when all characters
    align.

Time Complexity:  O(n * m)  — n = len(text), m = len(pattern)
Space Complexity: O(1)      — no auxiliary data structure needed

When to use:
    Short texts or patterns, situations where simplicity > performance.

DSA concept: Two-pointer / nested loop over arrays.
"""


def naive_search(text: str, pattern: str) -> list:
    """
    Find all starting indices where 'pattern' occurs in 'text'.

    Args:
        text:    The haystack string (full document text).
        pattern: The needle string (phrase to find).

    Returns:
        List of integer start-indices where the pattern is found.
        Returns an empty list if no match is found.

    Example:
        >>> naive_search("abcababc", "ab")
        [0, 3, 5]
    """
    matches = []          # result list — indices of each match
    n = len(text)         # length of haystack
    m = len(pattern)      # length of needle

    if m == 0 or m > n:
        return matches    # edge cases: empty pattern or pattern longer than text

    # Slide window across text
    for i in range(n - m + 1):           # O(n) outer loop
        match_found = True

        for j in range(m):               # O(m) inner comparison
            if text[i + j] != pattern[j]:
                match_found = False
                break                    # mismatch — stop inner loop early

        if match_found:
            matches.append(i)

    return matches


# ---------------------------------------------------------------------------
# Sentence-level naive matching
# ---------------------------------------------------------------------------

def naive_sentence_match(sentences_orig: list, sentences_sub: list) -> list:
    """
    Compare every sentence in 'sentences_sub' against every sentence in
    'sentences_orig' using exact naive string matching.

    Args:
        sentences_orig: List of sentences from the original document.
        sentences_sub:  List of sentences from the submitted document.

    Returns:
        List of tuples: (submitted_sentence, original_sentence, start_index)
        for every exact match found.
    """
    results = []

    for sub_sent in sentences_sub:
        sub_clean = sub_sent.lower().strip()

        for orig_sent in sentences_orig:
            orig_clean = orig_sent.lower().strip()

            # Use naive_search: look for sub_sent inside orig_sent
            indices = naive_search(orig_clean, sub_clean)

            if indices:
                results.append({
                    "submitted_sentence": sub_sent,
                    "matched_original":   orig_sent,
                    "match_indices":      indices,
                    "algorithm":          "Naive"
                })
                break  # found in original — no need to check further originals

    return results
