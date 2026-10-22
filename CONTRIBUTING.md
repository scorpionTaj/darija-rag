# Contributing

## Setup and checks

```bash
make setup && make corpus && make ingest
make lint test            # what CI runs (CI skips the torch extra; tests fake all models)
pre-commit install        # optional: ruff on every commit
```

## Adding a document to the corpus

1. Add its download to `scripts/fetch_corpus.sh`, using an official source, and add its line to
   `data/SHA256SUMS` (`shasum -a 256 data/raw/<file>`).
2. Arabic PDFs: name them `*_ar.pdf` so `--ocr '*_ar.pdf'` picks them up. After ingesting, open a
   few chunks and check that the words aren't reversed or scrambled.
3. Add questions for it (next section), then run `darija-rag eval` to confirm that the existing
   numbers don't regress. More chunks also means more distractors.
4. Update the corpus table in the README.

## Adding eval questions

Rules and schema: [docs/evaluation.md](docs/evaluation.md). In short: read the page first, label
every page that answers the question (both language versions for the Labour Code), write a short
reference answer with the article number, and keep Darija questions in everyday words.

## Changing retrieval or prompts

Report before/after numbers in the PR:

```bash
darija-rag eval eval/questions.jsonl --retriever dense-m3+rewrite+rerank --retriever <yours>
darija-rag eval-answers eval/questions.jsonl --sample 40 --retriever <yours>
```

A change that helps one language and hurts another needs a justification in the PR description.

## Code style

- Python 3.11+ syntax, ruff with a line length of 100.
- New retrievers implement `retrieve(query, k) -> [(chunk, score)]` and are registered in
  `factory.py`.
- Tests must run offline. Fake the LLM (`tests/conftest.py::FakeLLM`) and the models (pass
  `encoder=` / `model=`).
