# darija-rag

Ask Moroccan law questions the way people actually ask them, in **Darija, Arabizi
(`chhal mn sa3a khasni nkhdem?`), Modern Standard Arabic or French**, and get answers grounded in
the official Labour Code and Family Code, with page citations. When the documents don't contain
the answer, the system says so instead of guessing.

Everything runs locally: OCR, retrieval and reranking on your machine, and the LLM through any
OpenAI-compatible server (LM Studio, Bionic, Ollama, or your own gateway).

```
$ darija-rag ask "chhal mn sa3a khasni nkhdem f simana?"
(searched as: ما هو عدد ساعات العمل القانونية في الأسبوع؟)
F l'anachitaat ghayr l-fellahiya, mdat l-choghl l-3adiya hiya 44 sa3a f l-osbou3 [1][2].

Sources:
  [1] labour_code_ar.pdf p.59: ... المادة 184 تحدد في النشاطات غير الفلاحية مدة الشغل العادية ...
  [2] labour_code_fr.pdf p.73: ...

(44.1s, arabizi)
```

## Results

Retrieval on 180 answerable questions (see [docs/evaluation.md](docs/evaluation.md)). Cells show
Recall@10, with Recall@1 in brackets.

| Retriever | All | MSA | Darija | Arabizi | French | FR→AR cross-lingual |
|---|---|---|---|---|---|---|
| `bm25` (baseline) | 0.62 (0.39) | 0.97 | 0.30 | 0.14 | 0.59 | 0.00 |
| `dense` (e5-small) | 0.63 (0.32) | 0.96 | 0.25 | 0.14 | 0.68 | 0.00 |
| `dense-m3` (BGE-M3) | 0.76 (0.38) | 0.97 | 0.59 | 0.14 | 0.85 | 0.62 |
| `dense-m3+rerank` | 0.79 (0.61) | 1.00 | 0.61 | 0.19 | 0.93 | 0.77 |
| `bm25+rewrite` | 0.75 (0.43) | 0.97 | 0.57 | 0.67 | 0.59 | 0.00 |
| `dense-m3+rewrite` | 0.88 (0.43) | 0.97 | 0.75 | 0.86 | 0.85 | 0.62 |
| **`dense-m3+rewrite+rerank`** (default) | **0.95 (0.72)** | 1.00 | **0.93** | **0.86** | 0.93 | 0.77 |
| `hybrid-m3+rewrite+rerank` | 0.95 (0.72) | 1.00 | 0.95 | 0.86 | 0.90 | 0.69 |

What each stage contributes:
- **BGE-M3 instead of e5-small** doubles Darija recall and makes French questions about the
  Arabic-only Family Code answerable (0.00 → 0.62).
- **Rewriting Darija and Arabizi into MSA** with the LLM is the biggest single gain for dialect
  questions: Arabizi goes from 0.14 to 0.86. MSA and French skip this step and its latency.
- **The cross-encoder reranker** mostly improves the top rank: Recall@1 goes from 0.43 to 0.72.
- **Fusing BM25 back in** adds nothing once the reranker is in place, and it hurts cross-lingual
  questions, so the default is dense-only.

**Answers** (`gemma-4-12b-it-mlx` in LM Studio generating, same model judging). This is a
**partial run**: the first 41 questions in file order, all of them Family Code MSA and Darija. The
full 200-question run takes ~3 hours on a laptop (see [docs/deployment.md](docs/deployment.md)).

| Questions | n | Accuracy | Faithful | Citation hit | False abstention | p50 s |
|---|---|---|---|---|---|---|
| all | 41 | 0.95 | 0.98 | 0.88 | 0.00 | 37.1 |
| Darija | 15 | 0.87 | 1.00 | 0.67 | 0.00 | 37.2 |
| MSA | 26 | 1.00 | 0.96 | 1.00 | 0.00 | 36.5 |

The Arabizi, French and out-of-scope questions still need a representative run:
`darija-rag eval-answers eval/questions.jsonl --sample 40 --resume`, then `darija-rag report`.
The judge is the generator model, so read these scores as optimistic.

## Quickstart

```bash
brew install tesseract tesseract-lang          # Arabic OCR
make setup                                     # Python 3.13 venv with PyTorch and dev tools
make corpus                                    # 3 official PDFs, checksum-verified
make ingest                                    # OCR + chunking, ~3 min once, then cached
```

Start an LLM. In LM Studio, load a chat model (tested with `gemma-4-12b-it-mlx`), then go to the
Developer tab and click **Start Server**. Then:

```bash
.venv/bin/darija-rag ask "واش عندي الحق فنهار راحة فالسيمانة؟"
make serve                                     # chat UI at http://127.0.0.1:8000
```

The first run downloads BGE-M3 and the reranker from Hugging Face, about 4.5 GB.

### Docker

```bash
make corpus                                    # on the host
docker compose run --rm api darija-rag ingest --ocr '*_ar.pdf'
docker compose up --build                      # http://localhost:8000
```

The container reaches LM Studio on the host at `host.docker.internal:1234`.

## Usage

| Command | What it does |
|---|---|
| `darija-rag ingest --ocr '*_ar.pdf'` | extract, OCR and chunk `data/raw/` into `data/processed/chunks.jsonl` |
| `darija-rag search QUERY --retriever SPEC` | ranked chunks with scores, no LLM needed unless the spec has `+rewrite` |
| `darija-rag ask QUESTION` | cited answer or abstention |
| `darija-rag eval eval/questions.jsonl --retriever A --retriever B` | retrieval table by language and question type |
| `darija-rag eval-answers eval/questions.jsonl [--sample 40] [--resume]` | answer accuracy, faithfulness, citations, abstention, latency |
| `darija-rag report [answers.jsonl]` | answer table from finished rows, no LLM |
| `darija-rag serve` | API and chat UI |

**Retriever specs** are a base (`bm25`, `dense`, `dense-m3`, `hybrid`, `hybrid-m3`) plus optional
`+rewrite` and `+rerank` stages. The CLI, the API and the eval harness all accept them.

**API**

```bash
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "Combien de jours de congé pour une naissance ?", "k": 5}'
```

The response has `answer`, `abstained`, `variety` (msa / darija / arabizi / other),
`retrieval_query`, `citations` and `sources` (each with doc, page and text), and `timings`. Other
endpoints: `POST /api/search` and `GET /api/health`.

## Configuration

| Variable | Default | |
|---|---|---|
| `DARIJA_RAG_LLM_URL` | `http://localhost:1234/v1` | any OpenAI-compatible endpoint |
| `DARIJA_RAG_LLM_MODEL` | loaded LM Studio model | set it explicitly with other servers |
| `DARIJA_RAG_LLM_KEY` | – | bearer token, if needed |
| `DARIJA_RAG_JUDGE_MODEL` | generator model | a different judge gives stricter scores |
| `DARIJA_RAG_RETRIEVER` | `dense-m3+rewrite+rerank` | API default spec |
| `DARIJA_RAG_DEVICE` | `cpu` | `mps` or `cuda` for the embedding and reranking models |
| `DARIJA_RAG_CHUNKS` | `data/processed/chunks.jsonl` | |

`DARIJA_RAG_DEVICE` defaults to CPU on purpose. On a Mac where LM Studio runs the LLM on the GPU,
putting PyTorch on MPS as well made generation slower and retrieval latency erratic: 2–24 s, against
a steady ~7 s on CPU.

## Corpus

`scripts/fetch_corpus.sh` downloads from the Ministry of Justice portal (adala.justice.gov.ma) and
verifies `data/SHA256SUMS`.

| File | Text | Pages | Extraction |
|---|---|---|---|
| `labour_code_ar.pdf` | مدونة الشغل (Law 65.99), consolidated, uploaded July 2026 | 162 | OCR |
| `labour_code_fr.pdf` | Code du travail, consolidated to 26 October 2011 | 201 | text layer |
| `family_code_ar.pdf` | مدونة الأسرة (Law 70.03), 2021 update | 107 | OCR |

The Arabic PDFs' text layers are unusable: they swap lam-alef ligatures (مولانا comes out as موالنا)
and scramble letters inside words. Tesseract OCR gives clean text. The French code predates the
2027 reform bill (03-26).

## Project layout

```
src/darija_rag/   ingest, ocr, normalize, bm25, dense, retrieval, rerank, rewrite,
                  llm, answer, judge, factory, evaluate, api, cli, web/index.html
eval/             questions.jsonl (200 questions)
docs/             architecture, evaluation, api, deployment (backends, speed), troubleshooting
examples/         ask_api.py, batch_ask.py (questions.txt -> CSV), curl.sh
scripts/          fetch_corpus.sh
tests/            offline tests: models and the LLM are faked, CI runs without PyTorch
```

Start with [docs/architecture.md](docs/architecture.md) for the pipeline diagram and design decisions, [docs/api.md](docs/api.md) for the HTTP API, and [docs/troubleshooting.md](docs/troubleshooting.md) when something breaks. [CONTRIBUTING.md](CONTRIBUTING.md) covers adding documents or questions.

## Limitations

- **Latency.** On a 16 GB Mac, with Gemma 4 12B in LM Studio and the models on CPU, an answer takes
  about 30 s: ~7 s for retrieval and reranking, the rest for generation. The original < 2.5 s target
  needs a GPU server or a smaller model. Retrieval alone (`search`) without `+rerank` is under 1 s.
- **Eval-set bias.** One author wrote the questions, and the MSA ones share wording with the
  articles. The Darija detector was tuned on this set. The answer judge is the same model as the
  generator. See [docs/evaluation.md](docs/evaluation.md).
- **Legal currency.** The texts are the consolidated versions listed above. This is not legal
  advice.

## Roadmap

- [x] Ingestion with Arabic OCR, normalization, BM25 from scratch
- [x] Dense (e5, BGE-M3), hybrid RRF, cross-encoder reranking
- [x] 200-question eval set: MSA, Darija, Arabizi, French, cross-lingual, unanswerable
- [x] Darija/Arabizi → MSA query rewriting
- [x] Cited answers with abstention; LLM judge; latency
- [x] FastAPI service, chat UI, Docker, CI
- [ ] Answer in the question's dialect more reliably; Arabizi answers are inconsistent
- [ ] Bulletin Officiel issues and the Code des obligations et contrats
- [ ] Streaming answers in the UI
- [ ] A second annotator for the eval set
