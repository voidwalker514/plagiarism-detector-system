"""
report_generator.py
-------------------
Report Generation Module for Plagiarism Detector.

Generates:
  1. Console / terminal report (color-coded with colorama)
  2. Plain-text report file saved to outputs/
  3. JSON report file saved to reports/

DSA concept: Structured output using Python dicts (hash maps) + file I/O.
"""

import os
import json
import datetime

# Try to import colorama for colored output; gracefully degrade if not installed
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Create no-op stubs so the rest of the code works unchanged
    class _NoColor:
        def __getattr__(self, _): return ""
    Fore  = _NoColor()
    Back  = _NoColor()
    Style = _NoColor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color(text: str, color_code: str) -> str:
    """Apply a colorama color code if available."""
    if COLORAMA_AVAILABLE:
        return f"{color_code}{text}{Style.RESET_ALL}"
    return text


def _section(title: str, char: str = "=", width: int = 70) -> str:
    """Return a formatted section header string."""
    line = char * width
    return f"\n{line}\n  {title}\n{line}"


def _verdict_color(verdict: str) -> str:
    """Return verdict string with appropriate color."""
    if "ORIGINAL" in verdict:
        return _color(verdict, Fore.GREEN)
    elif "MINOR" in verdict:
        return _color(verdict, Fore.YELLOW)
    elif "MODERATE" in verdict:
        return _color(verdict, Fore.YELLOW)
    elif "HIGH" in verdict:
        return _color(verdict, Fore.RED)
    elif "SEVERE" in verdict:
        return _color(verdict, Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else "")
    return verdict


# ---------------------------------------------------------------------------
# Console Report
# ---------------------------------------------------------------------------

def print_console_report(
    orig_path:    str,
    sub_path:     str,
    scores:       dict,
    matched_sents: list,
    rk_result:    dict,
    kmp_matches:  list,
    ngram_size:   int,
    algorithm:    str,
) -> None:
    """
    Print a full, color-coded plagiarism analysis report to the terminal.

    Args:
        orig_path:     Path to original document.
        sub_path:      Path to submitted document.
        scores:        Dict from similarity.calculate_plagiarism_score().
        matched_sents: List of matched sentence dicts from naive_sentence_match().
        rk_result:     Dict from rabin_karp_ngram_similarity().
        kmp_matches:   List of matched phrase dicts from kmp_ngram_match().
        ngram_size:    N-gram window size used.
        algorithm:     Primary algorithm name used for display.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Header ──────────────────────────────────────────────────────────────
    print(_color(_section("PLAGIARISM DETECTION REPORT", "═"), Fore.CYAN))
    print(f"  Timestamp : {timestamp}")
    print(f"  Original  : {orig_path}")
    print(f"  Submitted : {sub_path}")
    print(f"  Algorithm : {algorithm}  |  N-gram size: {ngram_size}")

    # ── Scores ──────────────────────────────────────────────────────────────
    print(_color(_section("SIMILARITY SCORES"), Fore.CYAN))

    pct = scores["plagiarism_percentage"]
    bar_filled  = int(pct / 2)
    bar_empty   = 50 - bar_filled
    progress_bar = (
        _color("█" * bar_filled,  Fore.RED if pct > 50 else Fore.YELLOW)
        + _color("░" * bar_empty, Fore.WHITE)
    )
    print(f"\n  [{progress_bar}] {pct}%\n")

    print(f"  {'Metric':<35} {'Score':>10}")
    print(f"  {'─'*35} {'─'*10}")
    print(f"  {'N-gram Overlap (Rabin-Karp hash)':<35} {scores['ngram_overlap_score']:>9.2f}%")
    print(f"  {'Sentence Exact Match (Naive)':<35} {scores['sentence_match_score']:>9.2f}%")
    print(f"  {'Word Frequency Cosine':<35} {scores['word_frequency_score']:>9.2f}%")
    print(f"  {'Jaccard Token Similarity':<35} {scores['jaccard_score']:>9.2f}%")
    print(f"  {'─'*35} {'─'*10}")
    print(f"  {'PLAGIARISM PERCENTAGE (Weighted)':<35} {_color(f'{pct:>9.2f}%', Fore.RED if pct > 50 else Fore.YELLOW)}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(_color(_section("VERDICT"), Fore.CYAN))
    print(f"\n  {_verdict_color(scores['verdict'])}\n")

    # ── Matched Sentences (Naive) ────────────────────────────────────────────
    print(_color(_section(f"MATCHED SENTENCES — Naive Algorithm ({len(matched_sents)} found)"), Fore.CYAN))
    if matched_sents:
        for idx, m in enumerate(matched_sents[:10], 1):   # cap display at 10
            print(f"\n  [{idx}] {_color('SUBMITTED:', Fore.YELLOW)}  {m['submitted_sentence']}")
            print(f"       {_color('ORIGINAL: ', Fore.GREEN)}  {m['matched_original']}")
    else:
        print("  No exact sentence matches found.")

    # ── KMP N-gram Matches ───────────────────────────────────────────────────
    print(_color(_section(f"MATCHED PHRASES — KMP Algorithm ({len(kmp_matches)} found)"), Fore.CYAN))
    if kmp_matches:
        for idx, m in enumerate(kmp_matches[:10], 1):
            print(f"  [{idx}] \"{_color(m['phrase'], Fore.YELLOW)}\"")
    else:
        print("  No n-gram phrase matches found via KMP.")

    # ── Rabin-Karp Summary ───────────────────────────────────────────────────
    print(_color(_section("RABIN-KARP HASH ANALYSIS"), Fore.CYAN))
    print(f"  Matched N-grams   : {rk_result['match_count']} / {rk_result['total_sub']}")
    print(f"  Similarity (RK)   : {rk_result['similarity_pct']}%")
    if rk_result['matched_ngrams']:
        print(f"\n  Sample matched phrases (first 5):")
        for ph in rk_result['matched_ngrams'][:5]:
            print(f"    • \"{ph}\"")

    print(_color("\n" + "═" * 70 + "\n", Fore.CYAN))


# ---------------------------------------------------------------------------
# Save plain-text report
# ---------------------------------------------------------------------------

def save_text_report(
    orig_path:    str,
    sub_path:     str,
    scores:       dict,
    matched_sents: list,
    rk_result:    dict,
    output_dir:   str = "outputs",
) -> str:
    """
    Save a plain-text plagiarism report to the outputs/ directory.

    Returns:
        Path of the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(output_dir, f"report_{timestamp}.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  PLAGIARISM DETECTION REPORT")
    lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"  Original  : {orig_path}")
    lines.append(f"  Submitted : {sub_path}")
    lines.append("")
    lines.append("  SIMILARITY SCORES")
    lines.append("-" * 40)
    lines.append(f"  N-gram Overlap Score    : {scores['ngram_overlap_score']}%")
    lines.append(f"  Sentence Match Score    : {scores['sentence_match_score']}%")
    lines.append(f"  Word Frequency Score    : {scores['word_frequency_score']}%")
    lines.append(f"  Jaccard Score           : {scores['jaccard_score']}%")
    lines.append("-" * 40)
    lines.append(f"  PLAGIARISM PERCENTAGE   : {scores['plagiarism_percentage']}%")
    lines.append(f"  VERDICT                 : {scores['verdict']}")
    lines.append("")
    lines.append("  MATCHED SENTENCES (Naive)")
    lines.append("-" * 40)
    if matched_sents:
        for i, m in enumerate(matched_sents, 1):
            lines.append(f"  [{i}] SUBMITTED: {m['submitted_sentence']}")
            lines.append(f"       ORIGINAL : {m['matched_original']}")
            lines.append("")
    else:
        lines.append("  No exact sentence matches found.")
    lines.append("")
    lines.append("  RABIN-KARP SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Matched N-grams : {rk_result['match_count']} / {rk_result['total_sub']}")
    lines.append(f"  RK Similarity   : {rk_result['similarity_pct']}%")
    lines.append("=" * 70)

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return filename


# ---------------------------------------------------------------------------
# Save JSON report
# ---------------------------------------------------------------------------

def save_json_report(
    orig_path:    str,
    sub_path:     str,
    scores:       dict,
    matched_sents: list,
    rk_result:    dict,
    report_dir:   str = "reports",
) -> str:
    """
    Save a structured JSON report to the reports/ directory.

    Returns:
        Path of the saved JSON file.
    """
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(report_dir, f"report_{timestamp}.json")

    report = {
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "original_file": orig_path,
            "submitted_file": sub_path,
        },
        "scores": scores,
        "matched_sentences": [
            {
                "submitted": m["submitted_sentence"],
                "original":  m["matched_original"],
            }
            for m in matched_sents
        ],
        "rabin_karp": {
            "match_count":    rk_result["match_count"],
            "total_submitted": rk_result["total_sub"],
            "similarity_pct": rk_result["similarity_pct"],
            "sample_matched_phrases": rk_result["matched_ngrams"][:10],
        },
    }

    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)

    return filename
