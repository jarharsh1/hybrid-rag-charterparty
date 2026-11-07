"""Unit tests for BM25, the hybrid retriever, and query expansion."""

from charterqa.core.backends import HashingEmbedder
from charterqa.core.bm25 import BM25Index
from charterqa.core.pipeline import expand_query
from charterqa.core.retriever import HybridRetriever

CHUNKS = [
    "A customer may request a refund within 30 days of purchase.",
    "Priority incidents affecting production are acknowledged within one hour.",
    "Personal data is retained for 24 months after the last account activity.",
    "Passwords must be at least twelve characters and stored as salted hashes.",
]


def test_bm25_ranks_exact_term_first():
    idx = BM25Index()
    idx.add(CHUNKS)
    hits = idx.search("refund within days purchase", top_k=2)
    assert hits
    assert hits[0][0] == 0  # the refund chunk


def test_bm25_idf_non_negative():
    idx = BM25Index()
    idx.add(CHUNKS)
    assert idx._idf("the") >= 0


def test_hybrid_fuses_both_signals():
    emb = HashingEmbedder()
    r = HybridRetriever(emb, rrf_k=60)
    r.add(CHUNKS, source="kb.md")
    hits = r.search("how long to request a refund", top_k=2, candidate_k=4)
    assert hits
    top_chunk, score, ranks = hits[0]
    assert "refund" in top_chunk.text.lower()
    assert score > 0
    assert "dense_rank" in ranks or "bm25_rank" in ranks


def test_query_expansion_adds_domain_terms():
    expanded = expand_query("how do I get a refund")
    assert "return" in expanded or "refunds" in expanded


def test_empty_retriever_returns_nothing():
    emb = HashingEmbedder()
    r = HybridRetriever(emb)
    assert r.search("anything", top_k=3, candidate_k=5) == []
