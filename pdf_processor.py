from pathlib import Path
import fitz  # PyMuPDF
import re, os, json, numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
MODEL_PATH = "./english_minilm_model/"
INPUT_DIR = "/app/input/"
OUTPUT_DIR = "/app/output/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Load Semantic Model ---
try:
    model = SentenceTransformer(MODEL_PATH)
    heading_templates = [
        "introduction", "overview", "summary", "abstract", "conclusion", "references",
        "table of contents", "appendix", "background", "methodology", "results",
        "objectives", "acknowledgement", "requirements", "business plan",
        "system architecture", "training plan", "design considerations", "timeline",
        "executive overview", "data model", "implementation plan", "scope", "workflow",
        "roadmap", "user guide", "validation", "governance", "terms of reference"
    ]
    template_embeddings = model.encode(heading_templates, convert_to_tensor=True)
except Exception as e:
    model = None

# --- Helpers ---
normalize = lambda t: re.sub(r'\s+', ' ', t.strip().lower())
h1_pattern = r'^(Appendix [A-Z]:|[0-9]+\.)\s+'
h2_pattern = r'^[0-9]+\.[0-9]+\s+'
h3_pattern = r'^[0-9]+\.[0-9]+\.[0-9]+\s+'

def is_table_block(blk):
    """Detect table-like layout based on line count and vertical spacing."""
    lines = blk.get("lines", [])
    if len(lines) < 3:
        return False
    y_positions = [line["bbox"][1] for line in lines if "bbox" in line]
    if not y_positions:
        return False
    return (max(y_positions) - min(y_positions)) < 100

def compute_heading_score(block_embedding, block):
    s = 0.0
    text = block['text']
    avg_f = block.get('median_font_size', block['font_size'])

    if block['font_size'] / avg_f > 1.15: s += 0.30
    if block['bold']: s += 0.20
    if block['is_caps'] and block['word_count'] < 5: s += 0.15
    if block['ends_colon']: s += 0.10
    if block['word_count'] <= 10: s += 0.15
    if re.match(h1_pattern, text): s += 0.40
    if re.match(h2_pattern, text): s += 0.30
    if re.match(h3_pattern, text): s += 0.25

    if model and block_embedding is not None:
        s += 0.25 * util.cos_sim(block_embedding, template_embeddings).max().item()
    return round(min(s, 1.0), 3)

def classify(block, median_f):
    text = block['text']
    fs = block['font_size']
    wc = block['word_count']
    if re.match(h1_pattern, text): return "H1"
    if re.match(h2_pattern, text): return "H2"
    if re.match(h3_pattern, text): return "H3"

    sim_score = 0.0
    if model:
        emb = model.encode([text], convert_to_tensor=True)
        sim_score = util.cos_sim(emb, template_embeddings).max().item()

    if fs >= median_f * 1.35: return "H1"
    elif fs >= median_f * 1.20: return "H2"
    elif fs >= median_f * 1.10: return "H3"

    if wc <= 6 and block['is_caps']:
        if sim_score >= 0.5:
            return "H2"
    if sim_score >= 0.6:
        return "H2"
    return "H4"

def extract_page_blocks(page, pno, hf_to_remove):
    local_blocks = []
    block_data = page.get_text("dict")["blocks"]
    for blk in block_data:
        if blk.get("type", 0) != 0:
            continue
        is_table = is_table_block(blk)
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            txt = " ".join(s["text"].strip() for s in spans).strip()
            if len(txt) > 2 and txt not in hf_to_remove:
                local_blocks.append({
                    "text": txt,
                    "page": pno + 1,
                    "font_size": spans[0]["size"],
                    "bold": bool(spans[0]["flags"] & 2**4),
                    "is_caps": txt.isupper(),
                    "ends_colon": txt.endswith(":"),
                    "word_count": len(txt.split()),
                    "y_pos": spans[0]["bbox"][1],
                    "in_table": is_table
                })
    return local_blocks

# --- Main Function ---
def extract_document_outline(pdf_path: str):
    doc = fitz.open(pdf_path)

    # Header/footer removal
    hf_candidates = [b[4] for page in doc for b in page.get_text("blocks")
                     if b[1] < page.rect.height * 0.12 or b[3] > page.rect.height * 0.88]
    hf_to_remove = {text for text, count in Counter(hf_candidates).items() if count >= 2}

    # Parallel processing of blocks
    blocks = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_page_blocks, doc[pno], pno, hf_to_remove) for pno in range(len(doc))]
        for f in futures:
            blocks.extend(f.result())

    if not blocks:
        return {"title": "Document has no text content", "outline": []}

    median_f = np.median([b["font_size"] for b in blocks if b["word_count"] > 4])
    for b in blocks:
        b["median_font_size"] = median_f

    # Embeddings
    all_texts = [b['text'] for b in blocks]
    all_embeddings = model.encode(all_texts, convert_to_tensor=True) if model else [None] * len(blocks)

    # Title detection (from page 1)
    first_page_blocks = [b for b in blocks if b["page"] == 1]
    title = first_page_blocks[0]["text"] if first_page_blocks else ""
    if first_page_blocks:
        max_f = max(b["font_size"] for b in first_page_blocks)
        title_candidates = [b for b in first_page_blocks if b["font_size"] >= max_f * 0.9]
        if title_candidates:
            title = " ".join(c["text"] for c in sorted(title_candidates, key=lambda x: x["y_pos"]))
    t_norm = normalize(title)

    # Outline extraction
    outline, seen = [], set()

    def has_content_after(idx):
        for nb in blocks[idx+1: idx+10]:
            if nb["page"] != blocks[idx]["page"]: return False
            if nb["font_size"] <= blocks[idx]["font_size"]: return True
        return False

    for i, b in enumerate(blocks):
        if b.get("in_table"): continue
        n_txt = normalize(b["text"])
        if n_txt in seen or n_txt == t_norm: continue
        score = compute_heading_score(all_embeddings[i], b)
        if score > 0.60 and has_content_after(i):
            level = classify(b, median_f)
            outline.append({"level": level, "text": b["text"], "page": b["page"]})
            seen.add(n_txt)

    # Fix broken sequences like H3 after H1
    if len(outline) > 1:
        for i in range(1, len(outline)):
            cur = int(outline[i]["level"][1])
            prev = int(outline[i - 1]["level"][1])
            if cur - prev > 1:
                outline[i]["level"] = f"H{min(prev + 1, 3)}"

    return {"title": title, "outline": outline}

# --- Runner ---
def run_all(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        return

    for pdf_file in input_path.glob("*.pdf"):
        try:
            result = extract_document_outline(str(pdf_file))
            out_file = output_path / (pdf_file.stem + ".json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
        except Exception as e:
            pass # Suppress errors for minimal output

if __name__ == "__main__":
    run_all()
