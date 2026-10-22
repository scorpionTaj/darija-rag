# Changelog

## 0.2.0

- Eval set grown to 200 questions: Arabizi, cross-lingual (French → Arabic-only source), 20
  out-of-scope questions, several relevant pages per question, reference answers.
- BGE-M3 dense retrieval (`dense-m3`, `hybrid-m3`) and cross-encoder reranking
  (`bge-reranker-v2-m3`).
- Darija/Arabizi detection and LLM rewriting to MSA (`+rewrite`), cached on disk.
- Cited answer generation with localized abstention; LLM judge; latency metrics.
- `eval-answers` with `--sample`, `--resume`, `--no-judge`; `report` command.
- OpenAI-compatible LLM client (LM Studio, Bionic, Ollama, gateways), with LM Studio loaded-model
  detection.
- FastAPI service with a single-file RTL-aware chat UI.
- Dockerfile (CPU-only PyTorch, Tesseract with Arabic), Compose, GitHub Actions CI.
- Default retriever `dense-m3+rewrite+rerank`: Recall@10 0.95, up from 0.62 for BM25.

## 0.1.0

- Ingestion of the Labour Code (AR/FR) and Family Code from adala.justice.gov.ma, checksummed.
- Tesseract OCR for the Arabic PDFs, cached.
- Arabic normalization, BM25 from scratch, multilingual-e5 dense retrieval, RRF hybrid.
- 75-question retrieval eval split by language.
