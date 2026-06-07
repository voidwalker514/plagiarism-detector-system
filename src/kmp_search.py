"""
kmp_search.py
-------------
Knuth-Morris-Pratt (KMP) String Matching Algorithm.

Algorithm Description:
    KMP pre-processes the pattern to build a "failure function" (also called
    the Partial Match Table or LPS array — Longest Proper Prefix which is
    also a Suffix).  During the search phase, when a mismatch occurs, the
    failure function tells us how far to "fall back" in the pattern without
    re-examining characters in the text we have already passed.

Time Complexity:
    Pre-processing: O(m)   — building the LPS array
    Searching:      O(n)   — single pass through text
    Total:          O(n + m)   (much better than naive O(n*m))

Space Complexity: O(m)    — LPS array of size m

DSA concepts:
    - Array (LPS table)
    - Two-pointer technique (i for text, j for pattern)
    - Dynamic programming (LPS computation reuses previously computed values)

When to use:
    Large documents, repeated searches with the same pattern,
    when O(n+m) worst-case guarantee matters.
"""


# ---------------------------------------------------------------------------
# Step 1: Build the LPS (Longest Proper Prefix-Suffix) Array
# ---------------------------------------------------------------------------

def build_lps(pattern: str) -> list:
    """
    Compute the LPS (Longest Proper Prefix which is also Suffix) array
    for the given pattern.

    The LPS array is the heart of KMP.  lps[i] stores the length of the
    longest proper prefix of pattern[0..i] that is also a suffix.

    Example:
        pattern = "AAACAAAA"
        lps     = [0, 1, 2, 0, 1, 2, 3, 3]

    Args:
        pattern: The search pattern string.

    Returns:
        LPS array as a list of integers, same length as pattern.
    """
    m = len(pattern)
    lps = [0] * m        # initialize all values to 0

    length = 0           # length of previous longest prefix-suffix
    i = 1                # lps[0] is always 0; start from index 1

    while i < m:
        if pattern[i] == pattern[length]:
            # Characters match — extend the prefix-suffix
            length += 1
            lps[i] = length
            i += 1

        else:
            if length != 0:
                # Fall back using previously computed LPS value
                # (do NOT increment i here)
                length = lps[length - 1]
            else:
                # No prefix-suffix at position i
                lps[i] = 0
                i += 1

    return lps


# ---------------------------------------------------------------------------
# Step 2: KMP Search
# ---------------------------------------------------------------------------

def kmp_search(text: str, pattern: str) -> list:
    """
    Find all starting indices where 'pattern' occurs in 'text' using KMP.

    Args:
        text:    The haystack string (full document text).
        pattern: The needle string (phrase to find).

    Returns:
        List of integer start-indices where pattern is found in text.
        Returns an empty list if no match found.

    Example:
        >>> kmp_search("AABAACAADAABAABA", "AABA")
        [0, 9, 12]
    """
    matches = []
    n = len(text)
    m = len(pattern)

    if m == 0 or m > n:
        return matches

    # Pre-process: build LPS table  O(m)
    lps = build_lps(pattern)

    i = 0   # index for text
    j = 0   # index for pattern

    # Search phase  O(n)
    while i < n:
        if text[i] == pattern[j]:
            # Characters match — advance both pointers
            i += 1
            j += 1

        if j == m:
            # Full pattern matched — record start index
            matches.append(i - j)
            # Use LPS to find next possible match position
            j = lps[j - 1]

        elif i < n and text[i] != pattern[j]:
            # Mismatch after some matches
            if j != 0:
                # Fall back in pattern (don't move i)
                j = lps[j - 1]
            else:
                # No prefix matched — move text pointer forward
                i += 1

    return matches


# ---------------------------------------------------------------------------
# N-gram level KMP matching
# ---------------------------------------------------------------------------

def kmp_ngram_match(ngrams_orig: list, ngrams_sub: list) -> list:
    """
    Search for each n-gram phrase from 'ngrams_sub' inside the concatenated
    string of all original n-grams using KMP.

    Args:
        ngrams_orig: List of n-gram phrase strings from original document.
        ngrams_sub:  List of n-gram phrase strings from submitted document.

    Returns:
        List of dicts describing each matched phrase.
    """
    # Join original n-grams into a single searchable text (separator = '|')
    separator = " | "
    orig_text = separator.join(ngrams_orig)

    results = []

    for phrase in ngrams_sub:
        indices = kmp_search(orig_text.lower(), phrase.lower())
        if indices:
            results.append({
                "phrase":    phrase,
                "indices":   indices,
                "algorithm": "KMP"
            })

    return results
