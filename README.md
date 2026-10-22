# darija-rag

Question answering over Moroccan legal texts (Labour Code, Family Code, Bulletin Officiel) for
questions asked in **Modern Standard Arabic, Darija or French**, with source citations and a
measured evaluation harness.

This is **week 1**: ingestion, Arabic normalization, chunking, a BM25 search engine written from
scratch, and retrieval metrics. Every later stage (dense, hybrid, reranking, generation) is
compared against these BM25 numbers.

## Quickstart

```bash
uv venv && uv pip install -e '.[dev]'
# put PDFs or .txt files in data/raw/  (.txt: separate pages with form feeds \f)
.venv/bin/darija-rag ingest                     # -> data/processed/chunks.jsonl
.venv/bin/darija-rag search "العطلة السنوية" -k 5
.venv/bin/darija-rag eval eval/questions.jsonl  # recall@1/5/10, MRR@10, missed questions
.venv/bin/pytest
```

## Pipeline

| Stage     | Module                 | Notes                                                                                                                                                                   |
| --------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Extract   | `ingest.read_pages`  | PyMuPDF, 1-based page numbers kept for citations; NFKC folds Arabic presentation forms                                                                                  |
| Chunk     | `ingest.chunk_page`  | word windows (default 200 words, 50 overlap), never cross page boundaries                                                                                               |
| Normalize | `normalize.tokenize` | strips diacritics/tatweel; unifies alef variants, ى→ي, ة→ه, Arabic digits→ASCII; strips the article ال and its prefixes; drops stopwords (MSA, Darija, French) |
| Index     | `bm25.BM25`          | Okapi BM25 (k1=1.5, b=0.75) on an inverted index                                                                                                                        |
| Evaluate  | `evaluate.evaluate`  | a hit means a chunk from the labeled doc and page is in the top k                                                                                                       |

## Evaluation set

`eval/questions.jsonl`, one JSON object per line (see `eval/questions.example.jsonl`):

```json
{"question": "...", "doc": "file.pdf", "page": 12, "lang": "darija"}
```

Target: **200 questions**, split across MSA, Darija and French. Write each question from a page you
actually read, so the label is reliable. Report metrics per `lang`, since that split is the story of
the project.

The Darija gap shows up right away. On a three-page smoke test, BM25 finds the MSA questions, but
misses `شحال ديال الساعات كيخدم الأجير فالسيمانة؟` because no word in it matches `الأسبوع` or
`ساعة`. Dense retrieval (week 2) and Darija query rewriting (week 4) are meant to close that gap.

## Corpus

Sources, all public:

- Labour Code (Code du travail), Arabic and French versions
- Family Code (Moudawana)
- Selected Bulletin Officiel issues

After ingesting, check a few chunks in `data/processed/chunks.jsonl`. Some Arabic PDFs extract in
reversed visual order or with broken ligatures. If one does, re-export it as text or run OCR on it
before indexing.

## Roadmap

- [X] **Week 1:** ingestion, normalization, BM25 baseline, retrieval metrics
- [ ] **Week 1 (still to do):** collect the corpus, write the first 50 questions, record BM25 baseline numbers here, and try ablations (with/without article stripping, chunk size 100/200/400)
- [ ] **Week 2:** dense embeddings (BGE-M3 / multilingual-e5) in pgvector, hybrid BM25 + dense with reciprocal rank fusion, cross-encoder reranker; 200 questions
- [ ] **Week 3:** answer generation with citations through the LLM gateway (Ollama), faithfulness and hallucination scored by an LLM judge, p95 latency < 2.5 s
- [ ] **Week 4:** Darija query rewriting, FastAPI service, Next.js chat UI, Docker Compose, final results table

## Results

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --------- | -------- | -------- | --------- | ------ |
| BM25      | –       | –       | –        | –     |
