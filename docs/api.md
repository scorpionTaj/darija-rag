# HTTP API

Start it with `darija-rag serve` (http://127.0.0.1:8000) or `docker compose up`. FastAPI also serves
interactive docs at `/docs` and the OpenAPI schema at `/openapi.json`.

## `POST /api/ask`

Answer a question from the corpus, with citations, or abstain.

```json
{"question": "واش عندي الحق فنهار راحة فالسيمانة؟", "k": 5, "retriever": null}
```

| Field | Type | Default | |
|---|---|---|---|
| `question` | string, 1–1000 chars | – | MSA, Darija, Arabizi or French |
| `k` | int, 1–10 | 5 | sources given to the LLM |
| `retriever` | spec or null | `DARIJA_RAG_RETRIEVER` | for example `dense-m3+rewrite+rerank`, `bm25` |

Response (the shape is exact; the values are illustrative):

```json
{
  "answer": "إيه، عندك الحق فراحة أسبوعية ديال 24 ساعة على الأقل [1].",
  "abstained": false,
  "variety": "darija",
  "retrieval_query": "هل يحق للأجير الحصول على راحة أسبوعية؟",
  "retriever": "dense-m3+rewrite+rerank",
  "model": "gemma-4-12b-it-mlx",
  "citations": [{"n": 1, "id": "labour_code_ar.pdf#p65-c0", "doc": "labour_code_ar.pdf", "page": 65, "text": "..."}],
  "sources": [{"n": 1, "...": "..."}, {"n": 2, "...": "..."}],
  "timings": {"retrieve_s": 6.9, "generate_s": 24.1, "total_s": 31.0}
}
```

- `citations` are the sources the answer cites as `[n]`, a subset of `sources`.
- When `abstained` is true, `answer` holds a fixed "not in the documents" message in the
  question's language, and `citations` is empty.
- `retrieval_query` is the MSA rewrite for Darija and Arabizi questions, and the question itself
  otherwise.

## `POST /api/search`

Retrieval only. No LLM is needed unless the spec includes `+rewrite`.

```json
{"query": "مدة إجازة الولادة", "k": 5, "retriever": "dense-m3+rerank"}
```

The response has `retriever` and `results`, a list of `{id, doc, page, text, score}`. Scores are
comparable only within one retriever: BM25 scores, cosine similarity, RRF scores and cross-encoder
logits are on different scales.

## `GET /api/health`

```json
{"chunks": 818, "retriever": "dense-m3+rewrite+rerank",
 "llm": {"url": "http://localhost:1234/v1", "ok": true, "model": "gemma-4-12b-it-mlx"}}
```

If the LLM is unreachable, `llm.ok` is false and `llm.error` says why. If the corpus hasn't been
ingested, the endpoint returns 503.

## Errors

| Status | When |
|---|---|
| 422 | empty or oversized question, `k` out of range, unknown retriever spec |
| 503 | no `chunks.jsonl` yet; LLM endpoint unreachable or without a loaded model |

The `detail` field carries a human-readable message, which the chat UI displays as is.

## Notes

- The first request with a given spec loads its models (BGE-M3, the reranker), which can take
  10–60 s. Later requests reuse them. Model loading is serialized across concurrent requests.
- There is no authentication. Bind to `127.0.0.1` (the `serve` default), or put the API behind a
  reverse proxy before exposing it.
