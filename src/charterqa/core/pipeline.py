"""Hybrid RAG pipeline — domain-agnostic document Q&A.

Ingest any text corpus, then answer questions by fusing dense + BM25 retrieval
and generating a grounded answer with citations. An optional, configurable
query-expansion step bridges everyday wording and the exact terms used in your
documents; the default map is intentionally small and generic — extend
`EXPANSIONS` for your domain (see the README "Use cases").
"""

from __future__ import annotations

from charterqa.config import settings
from charterqa.core.backends import get_backends
from charterqa.core.chunking import chunk_text
from charterqa.core.retriever import HybridRetriever

# Small, generic synonym map bridging everyday wording and document terms.
# Extend this (or load your own) for a specific domain.
EXPANSIONS = {
    "refund": ["refunds", "return", "reimbursement"],
    "cancel": ["cancellation", "terminate"],
    "delete": ["deletion", "erasure", "remove"],
    "password": ["credentials", "login"],
    "overdue": ["late", "unpaid", "arrears"],
    "response": ["sla", "acknowledged", "first"],
}

_PROMPT = """You are a documentation assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say you don't know. Do
not invent facts.

CONTEXT:
{context}

QUESTION:
{question}
"""


def expand_query(question: str) -> str:
    extra: list[str] = []
    low = question.lower()
    for key, syns in EXPANSIONS.items():
        if key in low:
            extra.extend(syns)
    return question if not extra else f"{question} {' '.join(extra)}"


class CharterQaPipeline:
    def __init__(self, retriever: HybridRetriever | None = None):
        self.embedder, self.llm = get_backends()
        self.retriever = retriever or HybridRetriever(self.embedder, settings.rrf_k)

    def ingest(self, text: str, source: str) -> int:
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0
        self.retriever.add(chunks, source)
        return len(chunks)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        if len(self.retriever) == 0:
            return {"answer": "No contracts have been ingested yet.", "sources": []}

        k = top_k or settings.top_k
        q = expand_query(question)
        hits = self.retriever.search(q, top_k=k, candidate_k=settings.candidate_k)

        context = "\n\n".join(f"[{i + 1}] {c.text}" for i, (c, _, _) in enumerate(hits))
        answer = self.llm.generate(_PROMPT.format(context=context, question=question))

        sources = [
            {
                "rank": i + 1,
                "source": c.source,
                "fusion_score": round(score, 4),
                "retriever_ranks": rk,
                "excerpt": c.text[:200],
            }
            for i, (c, score, rk) in enumerate(hits)
        ]
        return {"answer": answer, "sources": sources}
