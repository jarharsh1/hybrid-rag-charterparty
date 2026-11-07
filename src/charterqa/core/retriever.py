"""Hybrid retriever: dense (cosine) + lexical (BM25), fused with RRF.

Reciprocal Rank Fusion combines the two rankings without needing to calibrate
their very different score scales: each document scores sum(1 / (rrf_k + rank))
across the retrievers that return it. A chunk that ranks highly on either
signal surfaces; one that ranks on both wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from charterqa.core.backends import Embedder
from charterqa.core.bm25 import BM25Index


@dataclass
class Chunk:
    id: int
    text: str
    source: str


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class HybridRetriever:
    def __init__(self, embedder: Embedder, rrf_k: int = 60):
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
        self._bm25 = BM25Index()

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, texts: list[str], source: str) -> None:
        embs = self.embedder.embed(texts)
        for text, emb in zip(texts, embs):
            self.chunks.append(Chunk(id=len(self.chunks), text=text, source=source))
            self._embeddings.append(emb)
        self._bm25.add(texts)

    def _dense(self, query: str, k: int) -> list[int]:
        q = self.embedder.embed([query])[0]
        scored = [(i, _cosine(q, e)) for i, e in enumerate(self._embeddings)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [i for i, s in scored[:k] if s > 0]

    def _lexical(self, query: str, k: int) -> list[int]:
        return [i for i, _ in self._bm25.search(query, k)]

    def search(self, query: str, top_k: int, candidate_k: int) -> list[tuple[Chunk, float, dict]]:
        dense = self._dense(query, candidate_k)
        lexical = self._lexical(query, candidate_k)

        fused: dict[int, float] = {}
        ranks: dict[int, dict] = {}
        for rank, idx in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
            ranks.setdefault(idx, {})["dense_rank"] = rank + 1
        for rank, idx in enumerate(lexical):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
            ranks.setdefault(idx, {})["bm25_rank"] = rank + 1

        ordered = sorted(fused.items(), key=lambda t: t[1], reverse=True)[:top_k]
        return [(self.chunks[i], score, ranks[i]) for i, score in ordered]
