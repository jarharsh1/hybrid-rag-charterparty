"""End-to-end: ingest the sample knowledge base via the API, then ask."""

from pathlib import Path

from fastapi.testclient import TestClient

from charterqa.api.main import app

client = TestClient(app)


def test_health_offline_backend():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"


def test_ingest_then_ask_refund():
    doc = Path("data/sample_knowledge_base.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "kb.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    r = client.post("/ask", json={"question": "How long do I have to request a refund?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"]
    assert "30 days" in body["answer"] or "30" in body["answer"]
    assert "retriever_ranks" in body["sources"][0]


def test_index_and_static_served():
    assert "Hybrid RAG Engine" in client.get("/").text
    assert "/ask" in client.get("/static/app.js").text
