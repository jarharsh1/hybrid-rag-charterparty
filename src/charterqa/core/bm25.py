"""A small, dependency-free BM25 lexical index.

BM25 shines on legal/contract text where exact terms matter — defined terms
("Laytime", "Demurrage", "Owners", "Charterers") and clause numbers that a
purely-semantic retriever can blur together.
"""

from __future__ import annotations

import math
from collections import Counter

from charterqa.core.backends import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.doc_len: list[int] = []
        self.avg_len: float = 0.0

    def add(self, texts: list[str]) -> None:
        for text in texts:
            toks = tokenize(text)
            self.docs.append(toks)
            self.doc_len.append(len(toks))
            for term in set(toks):
                self.doc_freq[term] += 1
        total = sum(self.doc_len)
        self.avg_len = total / len(self.docs) if self.docs else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        # BM25 idf with +1 smoothing (always non-negative).
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        # idf depends only on the index, not the document being scored, so compute
        # it once per distinct query term instead of for every (term, doc) pair.
        idf = {term: self._idf(term) for term in set(q_terms)}
        avg_len = self.avg_len or 1
        scores: list[tuple[int, float]] = []
        for i, toks in enumerate(self.docs):
            tf = Counter(toks)
            dl = self.doc_len[i] or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / avg_len)
                score += idf[term] * num / den
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]
