# 🔍 Plagiarism Detector Using String Matching Algorithms

![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Concepts](https://img.shields.io/badge/Concepts-String%20Matching%20%7C%20Hashing%20%7C%20Sliding%20Window-orange)

A complete industry-oriented Data Structures & Algorithms (DSA) project that detects text plagiarism by comparing a submitted document against an original source. This project implements three core string-matching algorithms from scratch without relying on heavy NLP libraries.

## 📌 Problem Statement

In academic and professional environments, verifying the originality of text is critical. Simple keyword searches are easily defeated by paraphrasing or minor edits. This project solves the problem by using sliding-window n-grams combined with advanced string-matching and hashing algorithms to detect both exact copies and heavy similarities efficiently.

## 🧠 DSA Concepts Demonstrated

1. **Knuth-Morris-Pratt (KMP)**: O(n+m) search using Longest Proper Prefix-Suffix (LPS) dynamic programming.
2. **Rabin-Karp Rolling Hash**: O(n+m) average time multi-pattern search using polynomial hashing.
3. **Sliding Window**: Used for N-gram phrase generation.
4. **Hash Sets (Hash Tables)**: O(1) membership lookups for fast n-gram intersection.
5. **Jaccard Similarity**: Mathematical metric for measuring set overlap.

## ⚙️ Features

- **Multi-Algorithm Engine**: Run Brute-Force, KMP, and Rabin-Karp to see how they perform.
- **N-gram Sliding Window**: Matches contiguous phrases rather than just isolated words.
- **Weighted Plagiarism Score**: Combines n-gram overlap (60%), sentence match (25%), and word frequency (15%).
- **Color-Coded Terminal Output**: Professional CLI experience with visual progress bars.
- **Automated Reporting**: Generates both human-readable `.txt` and machine-readable `.json` reports.

---

## 📂 Folder Structure

```text
Plagiarism-Detector-String-Matching/
│
├── documents/            # Sample text files (original.txt, submitted.txt)
├── src/                  # Source code modules
│   ├── preprocessor.py   # Text cleaning, tokenization, sliding window
│   ├── naive_search.py   # Brute-force exact matching O(n*m)
│   ├── kmp_search.py     # Knuth-Morris-Pratt algorithm O(n+m)
│   ├── rabin_karp.py     # Rolling hash algorithm O(n+m)
│   ├── similarity.py     # Mathematical scoring & verdicts
│   └── report_generator.py # Terminal & file output formatting
│
├── outputs/              # Plain-text plagiarism reports
├── reports/              # JSON plagiarism reports
├── images/               # Screenshots for GitHub documentation
├── docs/                 # In-depth DSA & Interview guides
├── main.py               # CLI entry point
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 🚀 How to Run

### 1. Setup & Installation
Ensure you have Python 3.8+ installed. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/Plagiarism-Detector-String-Matching.git
cd Plagiarism-Detector-String-Matching

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (colorama, tabulate)
pip install -r requirements.txt
```

### 2. Run the Demo
Run the project using the built-in sample documents to see it in action instantly:

```bash
python main.py --demo
```

### 3. Run with Your Own Files
Compare any two `.txt` files:

```bash
python main.py --original path/to/source.txt --submitted path/to/student.txt
```

### 4. Advanced Options
Adjust the n-gram size or select a specific algorithm:

```bash
python main.py --demo --ngram 4 --algorithm kmp
```
*Supported algorithms: `naive`, `kmp`, `rabin-karp`, `all`*

---

## 📊 Sample Output

*(Add a screenshot here using markdown image syntax: `![Terminal Output](images/output.png)`)*

**Similarity Scores:**
- N-gram Overlap (Rabin-Karp hash) : `65.50%`
- Sentence Exact Match (Naive) : `30.20%`
- Word Frequency Cosine : `85.10%`

**PLAGIARISM PERCENTAGE (Weighted)** : `59.62%`
**VERDICT**: 🔴 HIGH PLAGIARISM — Serious Concern (51-75%)

---

## 📚 In-Depth Documentation

Want to understand the math and logic behind this? Check out the `docs/` folder:
- [📖 Project & Algorithm Deep Dive](docs/PROJECT_EXPLANATION.md)
- [🧮 Algorithm Walkthroughs & Code Trace](docs/ALGORITHMS.md)
- [💼 Interview Preparation Guide](docs/INTERVIEW_PREP.md)

---

## 🎓 Learning Outcomes
By building this project, I gained practical experience in:
- Implementing theoretical string-matching algorithms (KMP, Rabin-Karp) in a real-world scenario.
- Managing algorithmic complexity (trading O(n*m) for O(n+m)).
- Designing a modular Python application architecture.
- Using mathematics (Jaccard, Cosine) for textual similarity analysis.

## 🤝 Let's Connect
If you found this project helpful for your DSA learning or interview prep, feel free to **Star** the repository!
