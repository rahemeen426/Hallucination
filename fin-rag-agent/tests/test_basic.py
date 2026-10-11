"""
Basic smoke tests -- checks the pure-logic parts of the code that don't need
Ollama running, so CI can run them without a live model.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import chunk_text, company_from_filename


def test_chunk_text_respects_size():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) > 1


def test_chunk_text_overlap():
    text = "abcdefghij" * 100
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert chunks[0][-150:] == chunks[1][:150]


def test_company_from_filename():
    assert company_from_filename("apple_2023_10k.pdf") == "apple"
    assert company_from_filename("tesla_2023_10k.pdf") == "tesla"


if __name__ == "__main__":
    test_chunk_text_respects_size()
    test_chunk_text_overlap()
    test_company_from_filename()
    print("All smoke tests passed.")
