# Changelog

## 0.4.0

- Commercial lease law 49.16 (Arabic) added: 8 documents, 2,293 chunks.
- Answer language check: when the answer is in a different language than the question (Darija,
  Arabizi or French), one rewrite pass fixes it. The fix is kept only if every `[n]` citation
  survives. New `language_match` column in `eval-answers`.
- Per-chunk embedding cache: adding a document embeds only its new chunks, not the whole corpus.
- Glossary: الحانوت, الساروت, السومة (47 entries). Darija answer markers (إيه, كاين, ماشي) added to
  the detector.
- Eval set: 282 questions (24 new on commercial leases).

## 0.3.0

- MIT license.
- Corpus grown from 3 to 7 official documents (818 → 2,260 chunks): Family Code in French, rental
  law 67.12 (Arabic), and the Code of Obligations and Contracts in Arabic and French.
- Darija/Arabizi → MSA legal glossary (`src/darija_rag/data/darija_glossary.tsv`, 44 entries).
  It extends dialect detection and adds term hints to the rewrite prompt. French loanwords
  (contrat, salaire) no longer count as Arabizi evidence.
- Rewrite cache tagged with a prompt/glossary id, so stale rewrites are discarded.
- Eval set: 258 questions. 58 are new (rental law, obligations and contracts), the 2 rent questions
  formerly out of scope now have answers, and every Family Code question also lists its French page.

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
