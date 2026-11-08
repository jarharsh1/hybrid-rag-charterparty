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


def test_bm25_idf_rare_term_outweighs_common_term():
    # idf must discriminate: a term in one document should score strictly higher
    # than a term in every document, and both must stay non-negative (+1 smoothing).
    idx = BM25Index()
    idx.add(
        [
            "common laytime demurrage",
            "common laytime cancellation",
            "common laytime arbitration",
        ]
    )
    rare_idf = idx._idf("demurrage")  # appears in exactly 1 of 3 docs
    common_idf = idx._idf("common")  # appears in all 3 docs
    assert rare_idf > common_idf
    assert common_idf >= 0
    assert idx._idf("missing") > rare_idf  # unseen term is rarest of all


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


def test_clear_resets_dense_and_bm25_state():
    emb = HashingEmbedder()
    r = HybridRetriever(emb)
    r.add(CHUNKS, source="kb.md")
    assert len(r) == len(CHUNKS)
    r.clear()
    assert len(r) == 0
    # The fused search returns nothing on an empty index.
    assert r.search("refund within days purchase", top_k=2, candidate_k=4) == []
    # BM25 must also be reset: a term from the dropped corpus is gone, and its
    # internal document list is empty (a stale index would corrupt later idf/tf).
    assert r._bm25.doc_freq.get("refund", 0) == 0
    assert r._bm25.docs == []
