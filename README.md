# darija-rag

Question answering over Moroccan legal texts (Labour Code, Family Code) for questions asked in
**Modern Standard Arabic, Darija or French**, with source citations and a measured evaluation
harness.

Status: **weeks 1–2 done**. OCR-based ingestion, Arabic normalization, a BM25 engine written from
scratch, dense retrieval (multilingual E5), hybrid fusion, and a 75-question eval set split by
language.

## Quickstart

```bash
brew install tesseract tesseract-lang         # Arabic OCR
uv venv --python 3.13 && uv pip install -e '.[dense,dev]'
./scripts/fetch_corpus.sh                     # 3 official PDFs from adala.justice.gov.ma, checksummed
.venv/bin/darija-rag ingest --ocr '*_ar.pdf'  # ~3 min the first time; OCR is cached
.venv/bin/darija-rag search "العطلة السنوية" -k 5 --retriever hybrid
.venv/bin/darija-rag eval eval/questions.jsonl --retriever bm25 --retriever dense --retriever hybrid
.venv/bin/pytest
```

Python 3.13 is pinned because PyTorch has no 3.15 wheels yet. BM25 alone needs neither PyTorch nor
the `dense` extra.

## Pipeline

| Stage | Module | Notes |
|---|---|---|
| Extract | `ingest.read_pages`, `ocr.ocr_pdf` | PyMuPDF text layer, or Tesseract OCR for files matching `--ocr`; 1-based page numbers kept for citations |
| Chunk | `ingest.chunk_page` | word windows (default 200 words, 50 overlap), never cross page boundaries |
| Normalize | `normalize.tokenize` | strips diacritics/tatweel; unifies alef variants, ى→ي, ة→ه, Arabic digits→ASCII; strips the article ال and its prefixes; drops stopwords (MSA, Darija, French) |
| Sparse | `bm25.BM25` | Okapi BM25 (k1=1.5, b=0.75) on an inverted index |
| Dense | `dense.DenseRetriever` | `intfloat/multilingual-e5-small` with `query:`/`passage:` prefixes, cosine over a cached NumPy matrix |
| Hybrid | `retrieval.HybridRetriever` | reciprocal rank fusion (k=60) over the top 50 of BM25 and dense |
| Evaluate | `evaluate.evaluate` | a hit means a chunk from the labeled doc and page is in the top k |

## Corpus

`scripts/fetch_corpus.sh` downloads these from the Ministry of Justice portal, adala.justice.gov.ma:

| File | Text | Pages | Extraction |
|---|---|---|---|
| `labour_code_ar.pdf` | مدونة الشغل (Law 65.99), consolidated, uploaded July 2026 | 162 | OCR |
| `labour_code_fr.pdf` | Code du travail, consolidated to 26 October 2011 | 201 | text layer |
| `family_code_ar.pdf` | مدونة الأسرة (Law 70.03), 2021 update | 107 | OCR |

**Why OCR the Arabic PDFs.** Their embedded text layers are unusable. The Family Code swaps the
lam-alef ligature (مولانا comes out as موالنا, الله as هللا), and the Labour Code scrambles letters
inside words. Tesseract (`ara`) gives clean text, apart from some punctuation noise: the Arabic
comma often comes out as `»` or as a `ء` glued to the previous word. Tesseract runs as a CLI
because PyMuPDF's built-in OCR reverses some right-to-left lines. Position sorting (`sort=True`)
is off for the same reason: it reverses Arabic word order.

The French code predates the reform bill 03-26, which is planned for 2027. Answers from it should
say that it is the 2011 consolidation.

## Evaluation set

`eval/questions.jsonl` has 75 questions: 36 MSA, 23 Darija and 16 French. Each one is written from a
page that was read and labeled by doc and page.

```json
{"question": "شحال من ساعة خاصني نخدم فالسيمانة؟", "doc": "labour_code_ar.pdf", "page": 59, "lang": "darija"}
```

**Caveat:** the MSA questions were written while reading the article, so they share vocabulary with
it. That favors BM25, so read the MSA numbers as an upper bound. The Darija questions use everyday
words on purpose (السيمانة, الكونطرا, الباطرون, جراو عليا).

## Results

Retrieval, 75 questions, chunks of 200 words with 50 overlap:

| Retriever | Questions | n | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | all | 75 | 0.47 | 0.68 | 0.76 | 0.56 |
| bm25 | darija | 23 | 0.09 | 0.26 | 0.30 | 0.14 |
| bm25 | fr | 16 | 0.38 | 0.69 | 0.88 | 0.52 |
| bm25 | msa | 36 | 0.75 | 0.94 | 1.00 | 0.84 |
| dense | all | 75 | 0.41 | 0.65 | 0.72 | 0.51 |
| dense | darija | 23 | 0.09 | 0.13 | 0.17 | 0.10 |
| dense | fr | 16 | 0.50 | 0.94 | 1.00 | 0.68 |
| dense | msa | 36 | 0.58 | 0.86 | 0.94 | 0.69 |
| **hybrid** | all | 75 | 0.47 | **0.75** | **0.80** | **0.58** |
| hybrid | darija | 23 | 0.04 | 0.26 | 0.39 | 0.13 |
| hybrid | fr | 16 | 0.44 | 1.00 | 1.00 | 0.67 |
| hybrid | msa | 36 | 0.75 | 0.94 | 0.97 | 0.82 |

BM25 ablations (Recall@10, all / darija):

| Variant | all | darija |
|---|---|---|
| baseline (200 words, article stripped) | 0.76 | 0.30 |
| without article stripping | 0.72 | 0.22 |
| chunks of 100 words | 0.73 | 0.30 |
| chunks of 400 words | 0.79 | 0.39 |

**Findings**
- Hybrid is the best retriever overall. Dense helps most on French paraphrases, and BM25 holds on
  exact MSA legal terms.
- Darija is the open problem. Every retriever stays under 0.40 Recall@10. Dense does worse than
  BM25 on it: only 76% of dense's top-5 chunks for Darija questions come from the right code,
  against 99% for MSA. e5-small does not understand Darija, so Darija→MSA query rewriting with the
  LLM is the next lever.
- Stripping the article helps Darija the most (Recall@10 0.22 → 0.30).
- Chunk size barely matters at 75 questions. Differences of one or two questions are noise.

## Roadmap

- [x] **Week 1:** ingestion, OCR, normalization, BM25 baseline, retrieval metrics, ablations
- [x] **Week 2:** dense retrieval (multilingual-e5-small), hybrid with reciprocal rank fusion
- [ ] **Week 2 (still to do):** cross-encoder reranker (`BAAI/bge-reranker-v2-m3`), compare BGE-M3 embeddings, grow the eval set to 200
- [ ] **Week 3:** answer generation with citations through the LLM gateway (Ollama), faithfulness and hallucination scored by an LLM judge, p95 latency < 2.5 s
- [ ] **Week 4:** Darija→MSA query rewriting, FastAPI service, Next.js chat UI, Docker Compose (pgvector once the corpus outgrows memory)
