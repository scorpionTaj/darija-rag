# API + chat UI. Corpus and indexes live in the mounted ./data volume, models in /models.
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only PyTorch: the default Linux wheel bundles CUDA (~2.5 GB) that this service never uses.
RUN uv pip install --system --no-cache \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    ".[dense]"

RUN useradd --create-home --uid 1000 app && mkdir -p /models /app/data && chown app /models /app/data
USER app
ENV HF_HOME=/models \
    DARIJA_RAG_CHUNKS=/app/data/processed/chunks.jsonl \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"
CMD ["uvicorn", "darija_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
