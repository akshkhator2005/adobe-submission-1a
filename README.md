
# PDF Outline Extractor

A lightweight Python-based tool to extract structured document outlines (titles and headings) from PDF files using [PyMuPDF](https://pymupdf.readthedocs.io/), [Sentence Transformers](https://www.sbert.net/), and semantic similarity.

## Features

* Extracts **document title** and **hierarchical outline (H1–H3)**.
* Removes repetitive headers/footers.
* Uses a **pre-downloaded SentenceTransformer model** for fast, CPU-only semantic matching.
* Processes PDFs in parallel using a **thread pool**.
* Outputs results as structured JSON.

## Project Structure

```
.
├── Dockerfile              # Builds the container
├── requirements.txt        # Lists base dependencies
├── download_model.py       # Downloads and caches the MiniLM model
├── pdf_processor.py        # Main PDF processing logic
├── input/                  # Place your PDFs here
└── output/                 # Extracted JSON outlines are saved here
```

## Setup

### 1. Build the Docker image

```bash
docker build -t pdf-outline-extractor .
```

### 2. Prepare input files

Create an `input/` directory in your project root and put your PDF files there:

```
mkdir -p input output
cp path/to/your/files/*.pdf input/
```

### 3. Run the container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-outline-extractor
```

All processed outlines will be available in the `output/` directory as `.json` files.

---

## How It Works

1. **Model download at build time**
   `download_model.py` downloads the [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) model into `./english_minilm_model/`.
2. **Semantic heading detection**

   * Scores each text block using font size, boldness, capitalization, and similarity to known heading templates.
   * Classifies headings into H1, H2, H3, and ignores repeated headers/footers.
3. **Output**

   * Saves JSON with structure:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Methodology", "page": 3}
  ]
}
```

---

## Requirements (if running locally without Docker)

* Python 3.10+
* Install dependencies:

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

Run:

```bash
python download_model.py
python pdf_processor.py
```

---
