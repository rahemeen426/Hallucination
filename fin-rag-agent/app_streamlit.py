"""
Streamlit frontend.

A simple chat-style UI to demo the agent: ask a question, see the answer,
which tools were used, the source filings, and two independent
hallucination-risk signals side by side:
  - LLM-judge groundedness (does the answer match the retrieved context?)
  - Self-consistency (does the answer stay stable across repeated sampling?)

Run locally (make sure the FastAPI backend is also running, see src/api.py):
    streamlit run app_streamlit.py
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/ask")

st.set_page_config(page_title="Financial Filings RAG Agent", page_icon="📊")

st.title("📊 Financial Filings RAG Agent")
st.caption("Ask about company 10-K filings, live stock prices, or both. Runs entirely on a local LLM via Ollama.")

if "history" not in st.session_state:
    st.session_state.history = []


def render_scores(data: dict):
    """Renders sources, groundedness, and self-consistency for one answer.
    `data` is either a history entry or the live `result` dict -- same shape."""
    if data.get("tools_used"):
        st.caption(f"Tools used: {', '.join(data['tools_used'])}")

    if data.get("sources"):
        with st.expander("Sources"):
            for s in data["sources"]:
                st.write(f"- {s['company']}, page {s['page']}")

    g = data.get("groundedness") or {}
    sc = data.get("selfcheck") or {}

    if g or sc:
        col1, col2 = st.columns(2)
        with col1:
            score = g.get("groundedness_score")
            verdict = g.get("verdict", "unknown")
            if score is not None:
                color = "green" if score >= 0.75 else "orange" if score >= 0.4 else "red"
                st.markdown(f"**LLM-judge groundedness**\n\n:{color}[{verdict}] ({score:.2f})")
        with col2:
            score = sc.get("consistency_score")
            verdict = sc.get("verdict", "unknown")
            if score is not None:
                color = "green" if score >= 0.85 else "orange" if score >= 0.65 else "red"
                st.markdown(f"**Self-consistency**\n\n:{color}[{verdict}] ({score:.2f})")

    if g.get("unsupported_claims"):
        st.warning(f"Unsupported claims flagged: {g['unsupported_claims']}")


# --- Replay chat history ---
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        render_scores(entry)

# --- Handle new question ---
question = st.chat_input("Ask a question, e.g. 'What was Apple's total revenue in 2023?'")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"question": question}, timeout=400)
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                st.error(f"Could not reach the API: {e}")
                st.stop()

        st.write(result["answer"])
        render_scores(result)

    st.session_state.history.append({"question": question, **result})
