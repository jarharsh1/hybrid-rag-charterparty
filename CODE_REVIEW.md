# Code Review — `hybrid-rag-charterparty`

A reviewer-facing deep-dive on the correctness and robustness findings fixed in
the second review round for the hybrid (BM25 + dense, RRF-fused) retriever, with
root causes, the actual before/after from each fix commit, and the regression
test that locks each one in.

Scope: the offline (CI/test) path — `HashingEmbedder` + BM25 + `ExtractiveLLM`.
Fixes keep the Ollama-backed path unchanged.

Commits reviewed: `72f6697`, `75e7cf8`, `4ed949d`.

---

## 1. BM25 idf recomputed per (query-term, document) — `src/charterqa/core/bm25.py:50`

**Where.** `BM25Index.search`, inside the nested `for term in q_terms` loop
that runs for every document.

**Root cause.** `idf` depends only on the index (corpus size and document
frequency), not on the document being scored, yet it was computed inside the
document loop:

```python
for i, toks in enumerate(self.docs):
    ...
    for term in q_terms:
        if term not in tf:
            continue
        idf = self._idf(term)      # recomputed for every (term, doc) pair
        ...
        score += idf * num / den
```

`_idf` does a `math.log` and dict lookups; with `D` documents and `Q` query
terms this is `O(D·Q)` log calls when `O(Q)` suffices. On a large corpus the
per-document re-derivation is wasted work, and it obscures that idf is a
per-term constant.

**Fix (`4ed949d`).** Hoist idf to a per-distinct-term dict computed once, and
cache `avg_len` locally:

```python
idf = {term: self._idf(term) for term in set(q_terms)}
avg_len = self.avg_len or 1
for i, toks in enumerate(self.docs):
    ...
    for term in q_terms:
        if term not in tf:
            continue
        num = tf[term] * (self.k1 + 1)
        den = tf[term] + self.k1 * (1 - self.b + self.b * dl / avg_len)
        score += idf[term] * num / den
```

**Correctness note.** Scores are numerically identical — this is a hoist, not a
formula change — so it is covered by the existing BM25 ranking/idf tests
(including `test_bm25_idf_rare_term_outweighs_common_term`) rather than a new one.

---

## 2. Startup pre-ingest silently did nothing off-root — `src/charterqa/api/main.py:24`

**Where.** Module load time, `_SAMPLE` definition.

**Root cause.** Same latent bug as its siblings: `_WEB_DIR` anchored to
`__file__`, but `_SAMPLE = Path("data/sample_knowledge_base.md")` was
CWD-relative, so the `if _SAMPLE.exists()` pre-ingest guard was false whenever
the server was launched off the repo root — empty index, no error.

**Fix (`72f6697`).**

```python
_ROOT = Path(__file__).resolve().parents[3]
_WEB_DIR = _ROOT / "web"
_SAMPLE = _ROOT / "data" / "sample_knowledge_base.md"
```

**Regression test** (`tests/test_api_e2e.py`):

```python
def test_sample_path_is_absolute_and_cwd_independent():
    assert api_main._SAMPLE.is_absolute()
    assert api_main._SAMPLE.exists()
```

---

## 3. `top_k=0` falsy-zero fallback — `src/charterqa/core/pipeline.py:70`

**Where.** `CharterQaPipeline.ask`.

**Root cause.** `k = top_k or settings.top_k` turns an explicit `top_k=0` into
the default, because `0` is falsy.

**Fix (`75e7cf8`).**

```python
k = settings.top_k if top_k is None else top_k
```

**Regression test** (`tests/test_retrieval.py`): drives it through the pipeline.

```python
def test_pipeline_top_k_zero_is_honoured_not_defaulted():
    p = CharterQaPipeline()
    p.reset()
    p.ingest(" ".join(CHUNKS), source="kb.md")
    result = p.ask("how long to request a refund", top_k=0)
    assert result["sources"] == []
```

---

## 4. Query expansion matched raw substrings — `src/charterqa/core/pipeline.py:43` (`expand_query`)

**Where.** `expand_query`, the `if key in low` test against the lowercased
question string.

**Root cause.** Expansion keys were matched as **substrings** of the whole
question:

```python
low = question.lower()
for key, syns in EXPANSIONS.items():
    if key in low:                 # substring match
        extra.extend(syns)
```

So `"response"` fires inside `"irresponsible"`, `"cancel"` inside
`"cancelled"`/`"cancellation"` of unrelated phrasing, etc. Every false hit
injects synonyms (`sla acknowledged first`, …) into the retrieval query,
polluting both the BM25 and dense signals. The failure mode **worsens as
`EXPANSIONS` grows**, which the docstring explicitly invites users to do — so
this is a scaling trap, not a cosmetic nit.

**Fix (`75e7cf8`).** Match whole tokens:

```python
words = set(tokenize(question))
for key, syns in EXPANSIONS.items():
    if key in words:               # whole-word match
        extra.extend(syns)
```

**Regression test** (`tests/test_retrieval.py`): proves the substring no longer
triggers while the standalone word still does.

```python
def test_query_expansion_matches_whole_words_not_substrings():
    assert expand_query("that was irresponsible conduct") == "that was irresponsible conduct"
    assert "sla" in expand_query("what is the response time")
```

---

## 5. Full sort for dense candidate selection — `src/charterqa/core/retriever.py:56` (`_dense`)

**Where.** `HybridRetriever._dense`.

**Root cause.** Scored all embeddings then `sort`ed the entire list to take the
top `k` candidates fed into RRF.

**Fix (`4ed949d`).** `heapq.nlargest` — `O(n log k)`, same top-k ordering:

```python
top = heapq.nlargest(k, scored, key=lambda t: t[1])
return [i for i, s in top if s > 0]
```

Pure efficiency; the `s > 0` filter and downstream RRF behavior are unchanged,
so the existing fusion tests cover it.

---

## Verified correct, not changed

- **BM25 idf** uses the standard `+1`-smoothed Lucene form
  `log(1 + (N - df + 0.5)/(df + 0.5))`, which is non-negative for every term and
  correctly ranks a rare term above a common one (asserted by
  `test_bm25_idf_rare_term_outweighs_common_term`).
- **`avg_len`** is recomputed over all documents on each `add`, so it stays
  correct across incremental ingests.
- **RRF fusion** (`retriever.search`) uses a consistent 0-based rank for both
  retrievers, sums `1/(rrf_k + rank)`, and dedups on chunk id; the chunk list and
  BM25 doc list stay index-aligned because both append in lockstep.
- **`HybridRetriever.clear()`** (prior round) rebuilds the BM25 index so stale
  document frequencies can't corrupt later idf/tf after a reset.
