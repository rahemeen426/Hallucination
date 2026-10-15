# Financial Filings RAG Agent

A local, agentic Retrieval-Augmented Generation system for financial document
Q&A, with **two independent hallucination-detection layers**. Runs entirely
on a local LLM via Ollama — no API costs, no data leaving your machine.

## What it does

Ask questions like:
- "What was Apple's total revenue in fiscal 2023?" → answered from the 10-K filing, with page citations
- "What is AAPL trading at right now?" → answered via a live stock-price lookup
- "How does Apple's 2023 revenue compare to Tesla's?" → retrieves both companies separately so neither gets crowded out
- "What was Apple's revenue growth from 2022 to 2023?" → calculates the change from reported statement figures and cites the filing page

Every answer generated from filings is scored two independent ways:
- **LLM-as-judge groundedness** — a separate model call checks whether the answer's claims are actually supported by the retrieved source text
- **Self-consistency (SelfCheckGPT-style)** — the answer is resampled several times at higher temperature; low agreement across samples signals the model may be confabulating rather than retrieving a stable fact

These two methods can disagree — that's expected and worth noting, since they catch different failure modes (one checks against evidence, the other checks internal stability).

## Project structure

```
fin-rag-agent/
├── data/
│   └── filings/           <- put your 10-K PDFs here
├── src/
│   ├── ingest.py           PDF -> chunks
│   ├── rag_core.py          embed, index, retrieve, generate
│   ├── agent.py              tool routing (filing search / stock price / calc)
│   ├── groundedness.py     hallucination check #1: LLM-as-judge
│   ├── selfcheck.py          hallucination check #2: self-consistency sampling
│   └── api.py                  FastAPI backend
├── app_streamlit.py        Streamlit demo UI (shows both scores side by side)
├── tests/test_basic.py     Smoke tests (no LLM required, safe for CI)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Setup

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure Ollama is running, and pull the models used:
   ```
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b
   ```

3. Add PDF 10-K filings to `data/filings/`, named like `apple_2023_10k.pdf`.
   If you add companies beyond Apple/Tesla, also update `KNOWN_COMPANIES` in
   `src/rag_core.py` and the ticker lookup in `src/agent.py`.

## Running it — command line

```
python src/ingest.py                          # 1. chunk the PDFs
python src/rag_core.py --build                 # 2. embed + index
python src/agent.py --ask "your question"        # 3. full agent: routing + both hallucination checks
```

## Running it — full app (API + web UI)

**Terminal 1:**
```
uvicorn src.api:app --reload --port 8000
```

**Terminal 2:**
```
streamlit run app_streamlit.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

To sanity-check the API alone without Streamlit, visit
http://localhost:8000/docs for an interactive test page.

## Running it — Docker

```
docker compose up --build
```
Ollama must still be running on your host machine.

## Running tests

```
python tests/test_basic.py
```

## Known limitations

- Local models (via Ollama) are smaller than hosted frontier models, so tool
  routing and both hallucination checks are less reliable than they'd be
  with larger models — worth naming explicitly in interviews/writeups.
- LLM-as-judge groundedness inherits the same model's blind spots — a model
  that hallucinates can also mis-judge its own hallucination. Self-consistency
  is included specifically to give a second, differently-biased signal.
- Self-consistency checking adds several extra LLM calls per question
  (`n_samples`, default 4), so answers take noticeably longer — an explicit
  accuracy/speed tradeoff, adjustable via the `n_samples` parameter.
- Ticker recognition in `agent.py` and company detection in `rag_core.py`
  use simple lookup tables — fine for a small demo, not a general solution.
- Financial calculations support annual changes for revenue/net sales, net
  income, operating income, and gross margin when both years appear in a
  retrieved statement table; other periods and ratios are not yet supported.
- Tool routing uses the LLM plus a keyword-based safety net (see
  `route()` in `agent.py`) because small local models are not fully
  reliable at structured classification on their own.
