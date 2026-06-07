"""
main.py
-------
Plagiarism Detector — Command-Line Entry Point

Usage examples:
  python main.py
  python main.py --original documents/original.txt --submitted documents/submitted.txt
  python main.py --original documents/original.txt --submitted documents/submitted.txt --ngram 6 --algorithm kmp
  python main.py --demo

Supported algorithms (--algorithm flag):
  naive        — Brute-force O(n*m), good for exact sentence matching
  kmp          — Knuth-Morris-Pratt O(n+m), optimal for pattern search
  rabin-karp   — Rolling hash O(n+m) avg, best for phrase-level hashing
  all          — Run all three and combine results (default)
"""

import argparse
import os
import sys

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path so src/ imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor     import preprocess
from src.naive_search     import naive_sentence_match
from src.kmp_search       import kmp_ngram_match
from src.rabin_karp       import rabin_karp_ngram_similarity, rabin_karp_phrase_match
from src.similarity       import calculate_plagiarism_score
from src.report_generator import (
    print_console_report,
    save_text_report,
    save_json_report,
)

# Try colorama for banner coloring
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    _C = True
except ImportError:
    class _NoColor:
        def __getattr__(self, _): return ""
    Fore  = _NoColor()
    Style = _NoColor()
    _C    = False


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
  ██████╗ ██╗      █████╗  ██████╗ ██╗ █████╗ ██████╗ ██╗███████╗███╗   ███╗
  ██╔══██╗██║     ██╔══██╗██╔════╝ ██║██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
  ██████╔╝██║     ███████║██║  ███╗██║███████║██████╔╝██║███████╗██╔████╔██║
  ██╔═══╝ ██║     ██╔══██║██║   ██║██║██╔══██║██╔══██╗██║╚════██║██║╚██╔╝██║
  ██║     ███████╗██║  ██║╚██████╔╝██║██║  ██║██║  ██║██║███████║██║ ╚═╝ ██║
  ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝

  ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗
  ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
  ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
  ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
  ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
  ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝

  String Matching Algorithms: Naive | KMP | Rabin-Karp
  DSA Course Project  ·  GitHub Portfolio  ·  v1.0.0
"""


def print_banner():
    if _C:
        print(Fore.CYAN + BANNER + Style.RESET_ALL)
    else:
        print(BANNER)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="plagiarism_detector",
        description="Plagiarism Detector using Naive, KMP, and Rabin-Karp algorithms.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--original", "-o",
        default="documents/original.txt",
        help="Path to the original (reference) document.\n"
             "Default: documents/original.txt",
    )
    parser.add_argument(
        "--submitted", "-s",
        default="documents/submitted.txt",
        help="Path to the submitted document to check.\n"
             "Default: documents/submitted.txt",
    )
    parser.add_argument(
        "--ngram", "-n",
        type=int,
        default=5,
        help="N-gram window size (number of words per phrase).\n"
             "Default: 5",
    )
    parser.add_argument(
        "--algorithm", "-a",
        choices=["naive", "kmp", "rabin-karp", "all"],
        default="all",
        help="String matching algorithm to use.\n"
             "  naive      – Brute-force O(n*m)\n"
             "  kmp        – Knuth-Morris-Pratt O(n+m)\n"
             "  rabin-karp – Rolling hash O(n+m) avg\n"
             "  all        – Run all three (default)",
    )
    parser.add_argument(
        "--save", "-sv",
        action="store_true",
        default=True,
        help="Save text and JSON reports to outputs/ and reports/.\n"
             "Default: True",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo documents (no file paths needed).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core Detection Pipeline
# ---------------------------------------------------------------------------

def run_detection(
    orig_path:  str,
    sub_path:   str,
    ngram_size: int      = 5,
    algorithm:  str      = "all",
    save_report: bool    = True,
) -> dict:
    """
    Execute the full plagiarism detection pipeline.

    Pipeline:
        1. Preprocess both documents
        2. Run Naive sentence-level matching
        3. Run KMP n-gram phrase matching
        4. Run Rabin-Karp hash-set similarity
        5. Calculate weighted aggregate score
        6. Print console report
        7. (Optionally) Save text + JSON reports

    Args:
        orig_path:   Path to original document.
        sub_path:    Path to submitted document.
        ngram_size:  N-gram window size.
        algorithm:   Which algorithm(s) to run.
        save_report: Whether to save report files.

    Returns:
        Dict with all results.
    """
    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  📄  Loading original document  : {orig_path}")
    orig_data = preprocess(orig_path,  ngram_size=ngram_size)
    print(f"  📄  Loading submitted document : {sub_path}")
    sub_data  = preprocess(sub_path,   ngram_size=ngram_size)

    print(f"\n  Original  → {len(orig_data['sentences'])} sentences, "
          f"{len(orig_data['tokens'])} tokens, {len(orig_data['ngrams'])} n-grams")
    print(f"  Submitted → {len(sub_data['sentences'])} sentences, "
          f"{len(sub_data['tokens'])} tokens, {len(sub_data['ngrams'])} n-grams")
    print(f"  {'─'*60}")

    # ── Step 2: Naive Sentence Matching ────────────────────────────────────
    if algorithm in ("naive", "all"):
        print("\n  🔍  Running Naive (Brute-Force) string matching...")
        matched_sents = naive_sentence_match(
            orig_data["sentences"], sub_data["sentences"]
        )
        print(f"      → {len(matched_sents)} exact sentence match(es) found.")
    else:
        matched_sents = []

    # ── Step 3: KMP N-gram Matching ────────────────────────────────────────
    if algorithm in ("kmp", "all"):
        print("\n  🔍  Running KMP (Knuth-Morris-Pratt) string matching...")
        kmp_matches = kmp_ngram_match(orig_data["ngrams"], sub_data["ngrams"])
        print(f"      → {len(kmp_matches)} n-gram phrase match(es) found.")
    else:
        kmp_matches = []

    # ── Step 4: Rabin-Karp Hash Similarity ────────────────────────────────
    if algorithm in ("rabin-karp", "all"):
        print("\n  🔍  Running Rabin-Karp (Rolling Hash) similarity analysis...")
        rk_result = rabin_karp_ngram_similarity(
            orig_data["ngrams"], sub_data["ngrams"]
        )
        print(f"      → {rk_result['match_count']}/{rk_result['total_sub']} "
              f"n-grams matched ({rk_result['similarity_pct']}%).")
    else:
        rk_result = {
            "matched_ngrams": [],
            "match_count": 0,
            "total_sub": len(sub_data["ngrams"]),
            "similarity_pct": 0.0,
        }

    # ── Step 5: Aggregate Score ────────────────────────────────────────────
    print("\n  📊  Calculating plagiarism score...")
    scores = calculate_plagiarism_score(
        ngrams_orig    = orig_data["ngrams"],
        ngrams_sub     = sub_data["ngrams"],
        tokens_orig    = orig_data["tokens"],
        tokens_sub     = sub_data["tokens"],
        sentences_orig = orig_data["sentences"],
        sentences_sub  = sub_data["sentences"],
    )

    # ── Step 6: Console Report ─────────────────────────────────────────────
    print_console_report(
        orig_path     = orig_path,
        sub_path      = sub_path,
        scores        = scores,
        matched_sents = matched_sents,
        rk_result     = rk_result,
        kmp_matches   = kmp_matches,
        ngram_size    = ngram_size,
        algorithm     = algorithm.upper(),
    )

    # ── Step 7: Save Reports ───────────────────────────────────────────────
    results = {
        "scores":         scores,
        "matched_sents":  matched_sents,
        "kmp_matches":    kmp_matches,
        "rk_result":      rk_result,
    }

    if save_report:
        txt_path = save_text_report(orig_path, sub_path, scores, matched_sents, rk_result)
        json_path = save_json_report(orig_path, sub_path, scores, matched_sents, rk_result)
        print(f"  💾  Text report saved  → {txt_path}")
        print(f"  💾  JSON report saved  → {json_path}\n")
        results["text_report"] = txt_path
        results["json_report"] = json_path

    return results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    print_banner()
    args = parse_args()

    # If --demo flag, use default document paths
    if args.demo:
        print("  [DEMO MODE] Using built-in sample documents.\n")
        args.original  = "documents/original.txt"
        args.submitted = "documents/submitted.txt"

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(project_root, args.original) \
        if not os.path.isabs(args.original) else args.original
    sub_path  = os.path.join(project_root, args.submitted) \
        if not os.path.isabs(args.submitted) else args.submitted

    # Validate files exist
    for path, label in [(orig_path, "original"), (sub_path, "submitted")]:
        if not os.path.exists(path):
            print(f"  ❌  Error: {label} file not found: {path}")
            sys.exit(1)

    # Run detection
    run_detection(
        orig_path   = orig_path,
        sub_path    = sub_path,
        ngram_size  = args.ngram,
        algorithm   = args.algorithm,
        save_report = args.save,
    )


if __name__ == "__main__":
    main()
