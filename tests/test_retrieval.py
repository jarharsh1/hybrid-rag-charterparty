"""Unit tests for BM25, the hybrid retriever, and query expansion."""

from charterqa.core.backends import HashingEmbedder
from charterqa.core.bm25 import BM25Index
from charterqa.core.pipeline import expand_query
from charterqa.core.retriever import HybridRetriever

CHUNKS = [
    "Demurrage shall be payable at USD 25,000 per day for time lost beyond laytime.",
    "Freight shall be paid at USD 18.50 per metric tonne within five banking days.",
    "Laytime for loading and discharging shall be 72 running hours.",
    "Dangerous goods may only be shipped with prior written consent under the IMDG Code.",
]


def test_bm25_ranks_exact_term_first():
    idx = BM25Index()
    idx.add(CHUNKS)
    hits = idx.search("demurrage rate per day", top_k=2)
    assert hits, "expected BM25 hits"
    assert hits[0][0] == 0  # the demurrage chunk


def test_bm25_idf_non_negative():
    idx = BM25Index()
    idx.add(CHUNKS)
    # A term in every doc still yields a non-negative score contribution.
    assert idx._idf("the") >= 0


def test_hybrid_fuses_both_signals():
    emb = HashingEmbedder()
    r = HybridRetriever(emb, rrf_k=60)
    r.add(CHUNKS, source="cp.md")
    hits = r.search("what is the demurrage rate", top_k=2, candidate_k=4)
    assert hits
    top_chunk, score, ranks = hits[0]
    assert "demurrage" in top_chunk.text.lower()
    assert score > 0
    # The winning chunk should be found by at least one retriever.
    assert "dense_rank" in ranks or "bm25_rank" in ranks


def test_query_expansion_adds_domain_terms():
    expanded = expand_query("what happens if unloading is late")
    assert "demurrage" in expanded  # from "late" and/or "unloading"
    assert "discharge" in expanded or "discharging" in expanded


def test_empty_retriever_returns_nothing():
    emb = HashingEmbedder()
    r = HybridRetriever(emb)
    assert r.search("anything", top_k=3, candidate_k=5) == []
