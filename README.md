# Adobe India Hackathon 2025 – Round 1A Submission

## 🔍 Challenge Overview

In Round 1A, we are tasked with *extracting a clean outline* from a PDF, including:
•⁠  ⁠*Title*
•⁠  ⁠*Headings* (H1, H2, H3) with levels and page numbers

Our solution must run *offline, be **fast* (≤10 sec for 50 pages), use ≤200MB model, and process multiple PDFs in ⁠ /app/input/ ⁠, outputting results in ⁠ /app/output/ ⁠.

---

## 🧠 Approach Summary

•⁠  ⁠*Text Extraction*: Used PyMuPDF to extract structured blocks from each PDF page.
•⁠  ⁠*Heading Detection*:
  - Font size, boldness, and position heuristics
  - Regex for numbered patterns (e.g., 1., 1.1, 1.1.1)
  - Sentence-BERT embeddings (MiniLM) for content-aware heading recognition
•⁠  ⁠*Title Detection*: Extracted from the first page using largest font sizes
•⁠  ⁠*Table Detection*: ⁠ is_table_block() ⁠ function detects dense tabular layouts to avoid misclassifying table rows as headings.
•⁠  ⁠*Parallelism*: ThreadPoolExecutor boosts speed by processing pages concurrently.

---

## 🛠️ Setup Instructions

1.⁠ ⁠Place your PDFs in the ⁠ input/ ⁠ directory.
2.⁠ ⁠Make sure your English MiniLM model (⁠ english_minilm_model/ ⁠) is downloaded and present.
3.⁠ ⁠Run the following command:

```bash
python3 main.py
