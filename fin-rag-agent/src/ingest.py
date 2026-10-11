"""
Load PDFs (10-K filings), chunk them, and keep track of where each
chunk came from (company, page) so answers can later cite sources.

Usage:
    python src/ingest.py
Expects PDFs placed in data/filings/, named like: apple_2023_10k.pdf
"""

import os
import re
import json
import fitz  # PyMuPDF

FILINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "filings")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")

CHUNK_SIZE = 800       # approx characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def extract_text_by_page(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window chunker over raw text."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def company_from_filename(filename: str) -> str:
    # expects e.g. "apple_2023_10k.pdf" -> "apple"
    return filename.split("_")[0]


def build_chunks() -> list[dict]:
    all_chunks = []
    if not os.path.isdir(FILINGS_DIR):
        raise FileNotFoundError(f"Put PDF filings in {FILINGS_DIR}")

    pdf_files = [f for f in os.listdir(FILINGS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {FILINGS_DIR}")

    for filename in pdf_files:
        path = os.path.join(FILINGS_DIR, filename)
        company = company_from_filename(filename)
        print(f"Processing {filename} ...")

        pages = extract_text_by_page(path)
        for page_num, page_text in pages:
            page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk in enumerate(page_chunks):
                all_chunks.append(
                    {
                        "id": f"{company}_p{page_num}_c{idx}",
                        "company": company,
                        "source_file": filename,
                        "page": page_num,
                        "text": chunk,
                    }
                )

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")
