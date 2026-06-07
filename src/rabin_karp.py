"""
rabin_karp.py
-------------
Rabin-Karp Rolling Hash String Matching Algorithm.

Algorithm Description:
    Rabin-Karp uses hashing to speed up pattern searching.
    Instead of comparing characters one-by-one (like Naive), it computes a
    hash of the pattern and slides a "rolling hash" window across the text.
    Only when the hashes match does it perform a character-by-character
    verification (to handle hash collisions).

    Rolling Hash Formula:
        hash(s[i+1..i+m]) = (hash(s[i..i+m-1]) - s[i] * BASE^(m-1)) * BASE + s[i+m]
        (mod PRIME to keep numbers manageable)

Time Complexity:
    Average: O(n + m)  — hash comparisons are O(1) with rolling update
    Worst:   O(n * m)  — if many hash collisions occur (rare with good primes)

Space Complexity: O(1)

DSA concepts:
    - Hashing (polynomial rolling hash)
    - Sliding Window (hash window slides across text)
    - Modular arithmetic (prevents integer overflow)

When to use:
    Multi-pattern search, large documents, when hashing overhead is acceptable.
    Great for detecting identical phrases across long corpora.
"""


# ---------------------------------------------------------------------------
# Constants for Rabin-Karp
# ---------------------------------------------------------------------------
BASE  = 256          # number of characters in the input alphabet (ASCII)
PRIME = 101          # a prime number used as modulus (reduces collisions)


# ---------------------------------------------------------------------------
# Core Rabin-Karp Search
# ---------------------------------------------------------------------------

def rabin_karp_search(text: str, pattern: str) -> list:
    """
    Find all starting indices where 'pattern' occurs in 'text'
    using the Rabin-Karp rolling-hash algorithm.

    Args:
        text:    The haystack string.
        pattern: The needle string.

    Returns:
        List of integer start-indices where pattern is found.

    Example:
        >>> rabin_karp_search("GEEKS FOR GEEKS", "GEEK")
        [0, 10]
    """
    matches = []
    n = len(text)
    m = len(pattern)

    if m == 0 or m > n:
        return matches

    # h = BASE^(m-1) mod PRIME  (highest positional weight)
    h = pow(BASE, m - 1, PRIME)

    # Compute initial hashes for pattern and first window of text
    pattern_hash = 0
    window_hash  = 0

    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % PRIME
        window_hash  = (BASE * window_hash  + ord(text[i]))    % PRIME

    # Slide the window across the text
    for i in range(n - m + 1):

        # --- Hash match: verify character by character (avoid false positives)
        if pattern_hash == window_hash:
            if text[i : i + m] == pattern:          # O(m) verification
                matches.append(i)

        # --- Roll the hash to the next window (O(1) update)
        if i < n - m:
            window_hash = (
                BASE * (window_hash - ord(text[i]) * h) + ord(text[i + m])
            ) % PRIME

            # Python mod can return negative values — normalize
            if window_hash < 0:
                window_hash += PRIME

    return matches


# ---------------------------------------------------------------------------
# Set-based n-gram similarity (HashSet intersection)
# ---------------------------------------------------------------------------

def rabin_karp_ngram_similarity(ngrams_orig: list, ngrams_sub: list) -> dict:
    """
    Compute similarity between two documents using a HashSet of n-grams.

    Algorithm:
        1. Hash all n-grams from original into a Python set  (O(k) time)
        2. For each n-gram from submitted, check set membership (O(1) avg)
        3. Count matches → compute Jaccard-style similarity

    DSA concept: Hash Set — O(1) average lookup via hash table.

    Args:
        ngrams_orig: List of n-gram phrases from the original document.
        ngrams_sub:  List of n-gram phrases from the submitted document.

    Returns:
        Dict with:
            'matched_ngrams'  – list of n-gram strings that matched
            'match_count'     – number of matched n-grams
            'total_sub'       – total n-grams in submitted doc
            'similarity_pct'  – similarity percentage (0-100)
    """
    # Build hash set of original n-grams (lowercase for case-insensitive compare)
    orig_set = {ng.lower() for ng in ngrams_orig}   # Python set = hash table

    matched = []
    for ng in ngrams_sub:
        if ng.lower() in orig_set:                   # O(1) average hash lookup
            matched.append(ng)

    total_sub   = len(ngrams_sub)
    match_count = len(matched)
    similarity  = (match_count / total_sub * 100) if total_sub > 0 else 0.0

    return {
        "matched_ngrams":  matched,
        "match_count":     match_count,
        "total_sub":       total_sub,
        "similarity_pct":  round(similarity, 2),
    }


# ---------------------------------------------------------------------------
# Rabin-Karp phrase-level search across corpus
# ---------------------------------------------------------------------------

def rabin_karp_phrase_match(ngrams_orig: list, ngrams_sub: list) -> list:
    """
    Search each submitted n-gram phrase against the original document text
    using the Rabin-Karp algorithm.

    Args:
        ngrams_orig: N-gram phrases from original document.
        ngrams_sub:  N-gram phrases from submitted document.

    Returns:
        List of dicts describing each matched phrase and its position(s).
    """
    separator = " | "
    orig_text  = separator.join(ngrams_orig)

    results = []
    for phrase in ngrams_sub:
        indices = rabin_karp_search(orig_text.lower(), phrase.lower())
        if indices:
            results.append({
                "phrase":    phrase,
                "indices":   indices,
                "algorithm": "Rabin-Karp"
            })

    return results
