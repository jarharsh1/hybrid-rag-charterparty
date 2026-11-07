"""Hybrid RAG pipeline for charter-party / contract Q&A.

Ingest contract text, then answer questions by fusing dense + BM25 retrieval
and generating a grounded answer with clause citations. A light query-expansion
step adds a few domain synonyms so colloquial questions still hit the defined
contract terms.
"""

from __future__ import annotations

from charterqa.config import settings
from charterqa.core.backends import get_backends
from charterqa.core.chunking import chunk_text
from charterqa.core.retriever import HybridRetriever

# Minimal domain synonym map to bridge everyday wording and contract terms.
_EXPANSIONS = {
    "late": ["demurrage", "delay"],
    "delay": ["demurrage"],
    "unloading": ["discharge", "discharging"],
    "loading": ["load", "laytime"],
    "waiting": ["laytime", "demurrage"],
    "payment": ["freight", "hire"],
    "owner": ["owners", "shipowner"],
    "renter": ["charterer", "charterers"],
}

_PROMPT = """You are a contracts assistant. Answer the question using ONLY the
charter-party context below. If the answer is not in the context, say you don't
know. Do not invent clause numbers.

CONTEXT:
{context}

QUESTION:
{question}
"""


def expand_query(question: str) -> str:
    extra: list[str] = []
    low = question.lower()
    for key, syns in _EXPANSIONS.items():
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
