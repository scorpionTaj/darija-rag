# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LLM endpoint ... is unreachable` | LM Studio server not started | LM Studio → Developer → Start Server, or set `DARIJA_RAG_LLM_URL` |
| `No models loaded` (HTTP 503) | server running, no model loaded | load a chat model in LM Studio, or run `~/.lmstudio/bin/lms load <model>` |
| Health shows an odd model (e.g. a TTS stub) | several models loaded, auto-pick chose the wrong one | set `DARIJA_RAG_LLM_MODEL` |
| `chunks.jsonl not found` | corpus not ingested | `make corpus && make ingest` |
| `tesseract is not installed` | OCR dependency missing | `brew install tesseract tesseract-lang` (Debian: `tesseract-ocr tesseract-ocr-ara`) |
| Arabic chunks read like `موالنا`, `هللا`, or reversed words | PDF ingested without OCR | `darija-rag ingest --ocr '*_ar.pdf'` |
| `torch` fails to install | Python 3.14+ has no PyTorch wheels yet | `uv venv --python 3.13` (pinned in `.python-version`) |
| Everything is slow, latency jumps around | memory swapping (`sysctl vm.swapusage`) | smaller LLM, close other apps; see [deployment.md](deployment.md) |
| `fetch_corpus.sh` checksum FAILED | the portal replaced a file | check the new PDF, then update `data/SHA256SUMS` and the eval page labels |
| Answer in MSA instead of Darija | the model drifted toward the language of the sources | expected sometimes; larger models follow the language rule better |
| An answerable question gets "not in the documents" | the relevant page was not retrieved | `darija-rag search "<question>" --retriever <spec>` to inspect; check `data/processed/rewrites.json` for a bad rewrite |

## Inspecting a wrong answer

```bash
darija-rag search "الولد فاش كيوصل خمسطاش العام" --retriever dense-m3+rewrite+rerank -k 10
grep '"q020"' data/processed/answers.jsonl | python3 -m json.tool
```

Each row in `answers.jsonl` holds the rewritten query, the answer, the cited pages and the verdict.
Most failures fall into one of three groups:
1. **Retrieval miss.** The relevant page is not in the top k.
2. **Bad rewrite.** The MSA rewrite lost a condition or a number.
3. **Generation.** The right page was there, but the answer misread it, or it was correct and the
   judge was wrong.

## Resetting caches

| Cache | Delete to rebuild |
|---|---|
| OCR text | `data/processed/ocr/` (~3 min) |
| Embeddings | `data/processed/embeddings/` (keyed by model and chunk content, so stale files are just unused) |
| Rewrites | `data/processed/rewrites.json` (re-asks the LLM) |
