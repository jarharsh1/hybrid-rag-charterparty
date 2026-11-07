# Hybrid RAG Engine — BM25 + Dense, RRF-fused

An **Ollama-first**, fully local, **domain-agnostic** Retrieval-Augmented
Generation engine. It answers questions over *any* text corpus by fusing
**lexical (BM25)** and **semantic (dense)** retrieval with **Reciprocal Rank
Fusion**, so both exact terms and paraphrases are matched — and every answer is
cited.

> Ships with a tiny neutral sample knowledge base so it runs out of the box.
> Replace `data/sample_knowledge_base.md` with your own corpus; nothing else
> changes.

## Why hybrid (the core idea)

Pure semantic search blurs look-alike wording; pure keyword search misses
paraphrases. Running both and fusing their rankings gets the best of each:

```
question ─► query expansion (optional synonyms)
          ├─► dense retriever  (embedding cosine) ──┐
          └─► BM25 retriever   (lexical)            ├─ RRF fusion ─► top-k ─► LLM ─► answer + citations
                                                    ┘
```

## Use cases

The engine is the same; only the corpus changes. Examples:

| Domain | Corpus you ingest | Example question |
|--------|-------------------|------------------|
| **Healthcare** | clinical protocols, guidelines | "What's first-line for type 2 diabetes?" |
| **Clinical trials** | study protocols, consent forms | "What are the exclusion criteria?" |
| **Legal** | contracts, charter parties, policies | "What's the demurrage rate on late discharge?" |
| **Customer support** | help center, FAQs, SLAs | "How long do I have to request a refund?" |
| **HR / IT** | employee handbook, runbooks | "What are the password requirements?" |
| **Finance / compliance** | regulations, filings | "What's the data-retention period?" |
| **Research** | papers, technical docs | "What method did the authors use for X?" |

Exact-term precision (BM25) is what makes this engine strong on **legal, medical,
and compliance** text, where a single defined term changes the meaning.

## Stack

- **Backend:** FastAPI (async Flask-equivalent; auto docs at `/docs`).
- **Frontend:** a browser chat UI (`web/`) at `/`, with per-citation fusion
  ranks. **This is where the user interacts.**
- **Model:** Ollama (local), with a deterministic offline fallback (so the
  pipeline and tests run with no model server — CI-safe).
- **Deploy:** single Docker image or `docker compose` (API + Ollama). See
  [DEPLOYMENT.md](DEPLOYMENT.md).

## Components

- **BM25** (`core/bm25.py`) — dependency-free lexical index.
- **Dense** — embedding cosine over the same chunks.
- **RRF** (`core/retriever.py`) — rank fusion without score calibration.
- **Query expansion** (`core/pipeline.py`) — a small, overridable synonym map
  (`EXPANSIONS`); extend it per domain.
- **API** (`api/main.py`) — `/`, `/health`, `/ingest`, `/ask`.

## Quickstart

```bash
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text   # optional
python demo.py                                         # or RAG_OFFLINE=1 python demo.py
uvicorn charterqa.api.main:app --reload                # UI at /, API at /docs
```

### Example

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How long do I have to request a refund?"}'
```

```json
{
  "answer": "A customer may request a refund within 30 days of purchase...",
  "sources": [
    { "rank": 1, "source": "sample_knowledge_base.md", "fusion_score": 0.03,
      "retriever_ranks": { "dense_rank": 2, "bm25_rank": 1 }, "excerpt": "..." }
  ]
}
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | models |
| `RAG_OFFLINE` | `0` | `1` forces the offline backend |
| `RAG_TOP_K` / `RAG_CANDIDATE_K` / `RAG_RRF_K` | `4` / `10` / `60` | retrieval + fusion |

## Tests

```bash
pytest -q          # offline; no Ollama needed
```

CI lints, runs tests + the demo offline, and builds the Docker image.
