# Charter-Party Contract QA — Hybrid RAG

An **Ollama-first**, fully local RAG service that answers questions over
**charter-party agreements** — laytime, demurrage, freight, notice of readiness,
clause lookup. A **Bahri-style maritime commercial use case**: chartering and
commercial teams need precise, cited answers from dense contract text where the
exact defined terms matter.

> Sample contract clauses are illustrative and fictional — not legal advice and
> not a real agreement.

## What makes it "hybrid"

Pure semantic (dense) retrieval blurs together look-alike legal language; pure
keyword (BM25) retrieval misses paraphrases. This service runs **both** and
fuses them with **Reciprocal Rank Fusion (RRF)**:

```
question ─► query expansion (domain synonyms)
          ├─► dense retriever  (embedding cosine) ──┐
          └─► BM25 retriever   (lexical)            ├─ RRF fusion ─► top-k ─► LLM ─► answer + citations
                                                    ┘
```

- **BM25** (`core/bm25.py`) — dependency-free lexical index; nails exact terms
  like *demurrage*, *laytime*, *Owners/Charterers* and clause numbers.
- **Dense** — embedding cosine over the same chunks (Ollama `nomic-embed-text`,
  or the deterministic offline embedder).
- **RRF** (`core/retriever.py`) — combines the two rankings without calibrating
  their score scales; each hit scores `sum(1 / (rrf_k + rank))`.
- **Query expansion** (`core/pipeline.py`) — maps everyday wording
  ("late", "unloading") to contract terms ("demurrage", "discharge").
- Each answer's sources expose the per-retriever ranks for transparency.

Ollama-first; falls back to a deterministic offline backend (hashed embedder +
extractive answerer) when Ollama is down, so the pipeline and tests run anywhere.

## Quickstart

```bash
pip install -r requirements.txt

ollama pull llama3.2 && ollama pull nomic-embed-text   # optional, for real answers
python demo.py                                         # or: RAG_OFFLINE=1 python demo.py

uvicorn charterqa.api.main:app --reload                # API at /docs
```

### Example

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is the demurrage rate if unloading is late?"}'
```

```json
{
  "answer": "Demurrage shall be payable ... at the rate of USD 25,000 per day ...",
  "sources": [
    { "rank": 1, "source": "voyage_charter.md", "fusion_score": 0.03,
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
| `RAG_TOP_K` | `4` | fused results returned |
| `RAG_CANDIDATE_K` | `10` | candidates per retriever before fusion |
| `RAG_RRF_K` | `60` | RRF constant |

## Tests

```bash
pytest -q          # offline; no Ollama needed
```

CI lints, runs tests + the demo offline, and builds the Docker image.
