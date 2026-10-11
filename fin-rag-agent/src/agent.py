"""
The agent layer.

Instead of always doing "retrieve -> answer" (plain RAG), the agent looks at
the question and decides which tool(s) it needs:

  - filing_search   -> looks up facts inside the 10-K PDFs (uses rag_core.py)
  - stock_price      -> looks up a live/recent stock price (yfinance)
  - calculate         -> does a simple arithmetic calculation

Routing is done by asking the LLM to classify the question, backed by a
keyword-based safety net -- small local models are not fully reliable at
this classification task, so obvious cases (e.g. "trading at", "stock
price") are force-included even if the LLM router misses them.

After answering, two independent hallucination-risk checks run:
  - groundedness.py  : LLM-as-judge, checks answer against retrieved context
  - selfcheck.py       : SelfCheckGPT-style, checks answer against resampled
                          versions of itself

Usage:
    python src/agent.py --ask "What was Apple's revenue in 2023 and what is AAPL trading at now?"
"""

import argparse
import json
import re
import ollama
import requests

try:
    from .selfcheck import selfcheck_consistency
    from .rag_core import answer_question as rag_answer_question
    from .groundedness import check_groundedness
except ImportError:  # pragma: no cover - direct script execution fallback
    from selfcheck import selfcheck_consistency
    from rag_core import answer_question as rag_answer_question
    from groundedness import check_groundedness

LLM_MODEL = "qwen2.5:3b"

ROUTER_PROMPT = """You are a routing assistant. Given a user question, decide
which tools are needed to answer it. Respond ONLY with a JSON list of tool
names, choosing from: "filing_search", "stock_price", "calculate".

Rules:
- Use "filing_search" for anything about a company's financial filings, revenue,
  risk factors, business description, or historical figures reported in a 10-K.
- Use "stock_price" for anything about a company's current/live stock price.
- Use "calculate" if the question requires arithmetic on numbers already
  mentioned or retrieved (e.g. ratios, differences, percentages).
- A question can need more than one tool.

Question: {question}

JSON list:"""


def route(question: str) -> list[str]:
    prompt = ROUTER_PROMPT.format(question=question)
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    raw = response["response"].strip()

    tools = []
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            valid = {"filing_search", "stock_price", "calculate"}
            tools = [t for t in parsed if t in valid]
        except json.JSONDecodeError:
            tools = []

    # Safety net: small local models are unreliable at this classification,
    # so obvious keyword cases are force-included regardless of what the
    # LLM router decided. This only adds tools a keyword strongly implies
    # are needed -- it never removes tools the LLM chose.
    lowered = question.lower()
    price_keywords = ["trading at", "stock price", "share price", "current price", "quote"]
    if any(k in lowered for k in price_keywords) and "stock_price" not in tools:
        tools.append("stock_price")

    filing_keywords = ["revenue", "filing", "10-k", "annual report", "risk factor", "net income", "reported", "compare"]
    if any(k in lowered for k in filing_keywords) and "filing_search" not in tools:
        tools.append("filing_search")

    return tools or ["filing_search"]  # safe default if nothing matched at all


def tool_filing_search(question: str) -> dict:
    result = rag_answer_question(question)
    return {
        "tool": "filing_search",
        "output": result["answer"],
        "sources": result["sources"],
        "context_text": result["context_text"],  # used by groundedness check
        "prompt": result["prompt"],                # used by selfcheck
    }


def tool_stock_price(company_ticker: str) -> dict:
    """company_ticker should be a ticker symbol, e.g. AAPL, TSLA."""
    try:
        response = requests.get(
            f"https://query2.finance.yahoo.com/v8/finance/chart/{company_ticker}",
            params={"range": "1d", "interval": "1m"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        chart = response.json().get("chart", {})
        if chart.get("error"):
            message = chart["error"].get("description", "Unknown provider error")
            return {"tool": "stock_price", "ticker": company_ticker, "output": message}
        results = chart.get("result") or []
        price = results[0].get("meta", {}).get("regularMarketPrice") if results else None
        if price is None:
            return {
                "tool": "stock_price",
                "ticker": company_ticker,
                "output": "No recent market price was returned.",
            }
        return {"tool": "stock_price", "ticker": company_ticker, "output": price}
    except (requests.RequestException, ValueError) as exc:
        return {
            "tool": "stock_price",
            "ticker": company_ticker,
            "output": f"Stock-price lookup failed: {exc}",
        }


def tool_calculate(expression: str) -> dict:
    """Very restricted eval -- digits, operators, parentheses, decimal points only."""
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return {"tool": "calculate", "output": "Invalid expression"}
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"tool": "calculate", "output": result}
    except Exception as e:
        return {"tool": "calculate", "output": f"Error: {e}"}


def extract_ticker(question: str) -> str | None:
    """Naive ticker extraction -- looks for known tickers or an ALL-CAPS token.
    Replace with a proper company->ticker lookup as you add more companies."""
    known = {"apple": "AAPL", "tesla": "TSLA"}
    lowered = question.lower()
    for name, ticker in known.items():
        if name in lowered:
            return ticker
    match = re.search(r"\b[A-Z]{2,5}\b", question)
    return match.group(0) if match else None


def run_agent(question: str) -> dict:
    tools_needed = route(question)
    tool_outputs = []

    for tool in tools_needed:
        if tool == "filing_search":
            tool_outputs.append(tool_filing_search(question))
        elif tool == "stock_price":
            ticker = extract_ticker(question)
            if ticker:
                tool_outputs.append(tool_stock_price(ticker))
        elif tool == "calculate":
            pass  # natural-language expression extraction: future improvement

    synthesis_prompt = f"""Question: {question}

Tool results:
{json.dumps(tool_outputs, indent=2, default=str)}

Using only the tool results above, write a clear, concise answer to the
question. If a tool result is missing or empty, say what's missing.

Answer:"""

    final = ollama.generate(model=LLM_MODEL, prompt=synthesis_prompt)
    final_answer = final["response"]

    # Both hallucination checks only run when filing data was actually used --
    # they're not meaningful for pure stock-price or calculation answers.
    filing_result = next((t for t in tool_outputs if t["tool"] == "filing_search"), None)
    groundedness = None
    selfcheck = None
    if filing_result:
        groundedness = check_groundedness(filing_result["context_text"], final_answer)
        selfcheck = selfcheck_consistency(filing_result["prompt"], final_answer, n_samples=2)

    return {
        "question": question,
        "tools_used": tools_needed,
        "tool_outputs": tool_outputs,
        "answer": final_answer,
        "groundedness": groundedness,
        "selfcheck": selfcheck,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, required=True)
    args = parser.parse_args()

    result = run_agent(args.ask)
    print("\nTOOLS USED:", result["tools_used"])
    print("\nANSWER:\n", result["answer"])
    if result["groundedness"]:
        print("\nGROUNDEDNESS (LLM-as-judge):\n", json.dumps(result["groundedness"], indent=2))
    if result["selfcheck"]:
        print("\nSELFCHECK (self-consistency):\n", json.dumps(result["selfcheck"], indent=2))
