.PHONY: setup corpus ingest test lint eval eval-answers serve docker

PY := .venv/bin
Q := eval/questions.jsonl

setup:  ## venv on Python 3.13 with dense models and dev tools
	uv venv --python 3.13 && uv pip install -e '.[dense,dev]'

corpus:  ## download the 3 official PDFs and verify checksums
	./scripts/fetch_corpus.sh

ingest:  ## OCR the Arabic PDFs (cached) and chunk everything
	$(PY)/darija-rag ingest --ocr '*_ar.pdf'

test:
	$(PY)/pytest

lint:
	$(PY)/ruff check . && $(PY)/ruff format --check .

eval:  ## retrieval comparison (no LLM needed)
	$(PY)/darija-rag eval $(Q) --retriever bm25 --retriever dense-m3 --retriever dense-m3+rerank \
		--retriever dense-m3+rewrite+rerank

eval-answers:  ## end-to-end answers, judged by the LLM (needs LM Studio running)
	$(PY)/darija-rag eval-answers $(Q)

serve:  ## API + chat UI on http://127.0.0.1:8000
	$(PY)/darija-rag serve

docker:
	docker compose up --build
