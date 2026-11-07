"""FastAPI serving layer for hybrid charter-party contract Q&A.

Endpoints:
  GET  /health   - liveness + active backend (ollama/offline)
  POST /ingest   - add contract text to the hybrid index
  POST /ask      - ask a question; fused dense+BM25 retrieval + grounded answer
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from charterqa.core.backends import OllamaLLM
from charterqa.core.pipeline import CharterQaPipeline

app = FastAPI(
    title="Charter-Party Contract QA (Hybrid RAG)",
    version="0.1.0",
    description="Ollama-first hybrid (BM25 + dense) RAG over charter-party "
    "agreements — laytime, demurrage, freight and clause lookup. Bahri-style "
    "maritime commercial use case.",
)

_pipeline = CharterQaPipeline()


class IngestRequest(BaseModel):
    text: str = Field(..., examples=["Demurrage shall be payable at USD 25,000 per day."])
    source: str = Field("inline", examples=["voyage-charter-2026.txt"])


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., examples=["What is the demurrage rate if unloading is late?"])
    top_k: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    backend = "ollama" if isinstance(_pipeline.llm, OllamaLLM) else "offline"
    return {"status": "ok", "backend": backend, "indexed_chunks": len(_pipeline.retriever)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    added = _pipeline.ingest(req.text, req.source)
    return IngestResponse(chunks_added=added, total_chunks=len(_pipeline.retriever))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**_pipeline.ask(req.question, req.top_k))
