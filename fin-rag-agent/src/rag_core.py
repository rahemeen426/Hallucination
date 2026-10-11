"""
The RAG core.

- Embeds chunks with Ollama's nomic-embed-text model
- Stores them in a local Chroma vector database
- Given a question, retrieves the top-k most relevant chunks
- Generates an answer with a local Ollama LLM, grounded in those chunks

Includes two fixes over the initial version:
  1. Multi-company retrieval: comparison questions ("Apple vs Tesla") now
     retrieve top-k chunks PER company, so one company can't crowd out the
     other in a single pooled similarity search.
  2. Terminology note in the prompt: some filings use "net sales" instead of
     "revenue" -- the LLM is told to treat these as equivalent.

Run once to build the index:
    python src/rag_core.py --build

Then test a query:
    python src/rag_core.py --ask "What was the revenue in 2023?"
"""

import os
import json
import argparse
import ollama
import chromadb

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"  # swap for whatever you pulled in Ollama

# Extend this list as you add more companies' filings to data/filings/
KNOWN_COMPANIES = ["apple", "tesla"]

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("filings")


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def build_index():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks (this may take a few minutes)...")
    for i, chunk in enumerate(chunks):
        vector = embed(chunk["text"])
        collection.add(
            ids=[chunk["id"]],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "company": chunk["company"],
                    "page": chunk["page"],
                    "source_file": chunk["source_file"],
                }
            ],
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(chunks)} embedded")

    print("Index built.")


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Plain pooled retrieval -- fine for single-company questions."""
    query_vector = embed(question)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"text": doc, "meta": meta, "distance": dist})
    return retrieved


def retrieve_multi_company(question: str, companies: list[str], top_k_each: int = 4) -> list[dict]:
    """Retrieves top_k_each chunks PER company, so no company gets crowded
    out of a comparison question by the other scoring slightly higher."""
    query_vector = embed(question)
    all_chunks = []
    for company in companies:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k_each,
            where={"company": company},
        )
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            all_chunks.append({"text": doc, "meta": meta, "distance": dist})
    return all_chunks


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['meta']['company']}, page {c['meta']['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    return f"""You are a financial analyst assistant. Answer the question using
ONLY the context below. Note that companies use different terms for revenue
(e.g. Apple's filings say "net sales") -- treat these as equivalent to
revenue unless the context specifically distinguishes them. If the context
does not contain the answer, say so clearly instead of guessing. Cite the
company and page for any figure you use.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, top_k: int = 5) -> dict:
    mentioned = [c for c in KNOWN_COMPANIES if c in question.lower()]

    if len(mentioned) >= 2:
        retrieved_chunks = retrieve_multi_company(question, mentioned)
    else:
        retrieved_chunks = retrieve(question, top_k=top_k)

    prompt = build_prompt(question, retrieved_chunks)

    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    answer_text = response["response"]

    context_text = "\n\n".join(c["text"] for c in retrieved_chunks)

    return {
        "question": question,
        "answer": answer_text,
        "prompt": prompt,          # used by selfcheck.py for resampling
        "context_text": context_text,  # used by groundedness.py
        "sources": [
            {"company": c["meta"]["company"], "page": c["meta"]["page"]}
            for c in retrieved_chunks
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the vector index from chunks.json")
    parser.add_argument("--ask", type=str, help="Ask a question against the index")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.ask:
        result = answer_question(args.ask)
        print("\nANSWER:\n", result["answer"])
        print("\nSOURCES:\n", result["sources"])
    else:
        parser.print_help()
