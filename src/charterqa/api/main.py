"""FastAPI serving layer for hybrid clinical-trial protocol Q&A.

Endpoints:
  GET  /         - the browser chat UI (static)
  GET  /health   - liveness + active backend (ollama/offline)
  POST /ingest   - add protocol text to the hybrid index
  POST /ask      - ask a question; fused dense+BM25 retrieval + grounded answer
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from charterqa.core.backends import OllamaLLM
from charterqa.core.pipeline import CharterQaPipeline

_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
_SAMPLE = Path("data/sample_knowledge_base.md")

app = FastAPI(
    title="Hybrid RAG Engine (BM25 + dense)",
    version="0.1.0",
    description="Ollama-first, domain-agnostic hybrid retrieval-augmented QA over "
    "any text corpus. Fuses lexical (BM25) and semantic (dense) retrieval so both "
    "exact terms and paraphrases are matched. Swap in your own documents.",
)

_pipeline = CharterQaPipeline()

# Pre-ingest the bundled synthetic protocol so the UI works immediately.
if _SAMPLE.exists():
    try:
        _pipeline.ingest(_SAMPLE.read_text(encoding="utf-8"), source=_SAMPLE.name)
    except Exception as exc:  # pragma: no cover - startup best-effort
        print(f"[startup] could not ingest {_SAMPLE}: {exc}", file=sys.stderr)


class IngestRequest(BaseModel):
    text: str = Field(..., examples=["Refunds are issued to the original payment method within five business days."])
    source: str = Field("inline", examples=["handbook.txt"])


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., examples=["How long do I have to request a refund?"])
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


# --- Browser UI ------------------------------------------------------------ #
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_WEB_DIR / "index.html"))
