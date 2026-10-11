"""
Step 6a: FastAPI backend.

Exposes the agent as a simple HTTP API so it can be called from the
Streamlit frontend, curl, Postman, or anything else.

Run locally:
    uvicorn src.api:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What was Apple's revenue in 2023?\"}"
"""

from fastapi import FastAPI
from pydantic import BaseModel

if __package__:
    from .agent import run_agent
else:  # pragma: no cover - direct script execution fallback
    from agent import run_agent

app = FastAPI(
    title="Financial Filings RAG Agent API",
    description="Ask questions about company 10-K filings, live stock prices, and calculations.",
    version="0.1.0",
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    tools_used: list[str]
    answer: str
    sources: list[dict] | None = None
    groundedness: dict | None = None
    selfcheck: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    result = run_agent(request.question)

    sources = []
    for tool_output in result["tool_outputs"]:
        if tool_output["tool"] == "filing_search":
            sources = tool_output.get("sources", [])

    return AnswerResponse(
        question=result["question"],
        tools_used=result["tools_used"],
        answer=result["answer"],
        sources=sources,
        groundedness=result["groundedness"],
        selfcheck=result["selfcheck"],
    )
