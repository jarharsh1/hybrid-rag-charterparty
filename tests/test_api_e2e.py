"""End-to-end: ingest the sample charter party via the API, then ask."""

from pathlib import Path

from fastapi.testclient import TestClient

from charterqa.api.main import app

client = TestClient(app)


def test_health_offline_backend():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"


def test_ingest_then_ask_demurrage():
    doc = Path("data/voyage_charter.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "voyage_charter.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    r = client.post("/ask", json={"question": "What is the demurrage rate if unloading is late?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"]
    assert body["sources"][0]["source"] == "voyage_charter.md"
    # The demurrage figure should surface via hybrid retrieval + extraction.
    assert "25,000" in body["answer"] or "25000" in body["answer"]
    # Fusion metadata is exposed for transparency.
    assert "retriever_ranks" in body["sources"][0]
