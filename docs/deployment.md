# Deployment and performance

## LLM backends

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint works. Point
`DARIJA_RAG_LLM_URL` at it.

| Backend | `DARIJA_RAG_LLM_URL` | Notes |
|---|---|---|
| LM Studio | `http://localhost:1234/v1` (default) | Developer tab → Start Server. The loaded model is picked automatically. |
| Bionic | its OpenAI-compatible URL | set `DARIJA_RAG_LLM_MODEL` |
| Ollama | `http://localhost:11434/v1` | `DARIJA_RAG_LLM_MODEL=qwen2.5:7b`, for example |
| Your gateway, vLLM, OpenRouter | gateway URL | set `DARIJA_RAG_LLM_KEY` if it needs a token |

From Docker, use `host.docker.internal` instead of `localhost`. `compose.yaml` does this by default.

**Model choice.** The model has to read MSA legal text and write Darija. Tested:
`gemma-4-12b-it-mlx` (LM Studio, MLX). Reasoning models also work, since `<think>` blocks are
stripped, but the hidden reasoning makes every answer slower.

## Why an answer takes ~30 s on a laptop, and how to make it faster

Measured on a 16 GB Apple Silicon Mac, with Gemma 4 12B in LM Studio and PyTorch models on CPU:

| Stage | Time | What happens |
|---|---|---|
| Rewrite (Darija/Arabizi only) | 2–5 s, 0 when cached | 1 short LLM call |
| Dense retrieval | < 0.5 s | encode the query with BGE-M3, then one matrix product |
| Rerank | ~6 s | 30 (query, chunk) pairs through a 568M-parameter cross-encoder |
| Generation | 20–30 s | ~2,500 prompt tokens (5 chunks of 200 words) plus the answer |

`eval-answers` adds an LLM judge call per question, about 14 s, so the full 200-question run takes
around 50 s per question, close to 3 hours. To iterate faster:

```bash
darija-rag eval-answers eval/questions.jsonl --sample 40          # every language and type, ~35 min
darija-rag eval-answers eval/questions.jsonl --resume             # continue a stopped run
darija-rag eval-answers eval/questions.jsonl --no-judge           # abstention and citations only
darija-rag report                                                 # table from finished rows, no LLM
```

Levers, from largest effect to smallest:
1. **A faster LLM.** A 3–4B model cuts generation by about 3×, at some cost in Darija quality. A
   GPU server running vLLM brings it to 1–2 s.
2. **Fewer sources** (`k=3`). This shortens the prompt. Recall@5 is 0.92 and Recall@3 is lower, so
   measure the effect before switching.
3. **Skip the reranker** (`dense-m3+rewrite`). This saves ~6 s, but Recall@1 drops from 0.72 to 0.43.
4. **`DARIJA_RAG_DEVICE=mps`.** This helps only when the LLM is *not* on the same GPU. With LM Studio
   on the same Mac it made things slower, because both compete for the GPU and memory.

**Memory.** BGE-M3 (2.3 GB), the reranker (2.3 GB) and a 12B LLM (~7 GB) together fill a 16 GB
machine, and swapping makes everything erratic. Close other heavy apps, or use a smaller LLM.

## Docker

```bash
./scripts/fetch_corpus.sh                                      # host: curl + shasum
docker compose run --rm api darija-rag ingest --ocr '*_ar.pdf' # container: Tesseract ara
docker compose up --build -d
```

- The image is about 1.8 GB (Python 3.13 slim, CPU-only PyTorch, Tesseract with Arabic) and runs
  as a non-root user.
- `./data` is mounted at `/app/data`, so chunks, OCR cache, embeddings and the rewrite cache persist.
- Hugging Face models are cached in the `models` volume. The first request downloads about 4.5 GB.
- Settings come from `.env`. Copy `.env.example`.

## Exposing it

The API has no authentication or rate limiting. For anything beyond your own machine, put it behind
a reverse proxy that adds both, and keep in mind that every `/api/ask` call costs an LLM generation.
