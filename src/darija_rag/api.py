"""HTTP API and chat UI.

    GET  /                 chat UI
    GET  /api/health       chunk count, retriever, LLM endpoint status
    POST /api/search       {"query", "k", "retriever"} -> ranked chunks
    POST /api/ask          {"question", "k", "retriever"} -> cited answer or abstention

Environment: DARIJA_RAG_CHUNKS (default data/processed/chunks.jsonl), DARIJA_RAG_RETRIEVER
(default dense-m3+rewrite+rerank) and the DARIJA_RAG_LLM_* variables of darija_rag.llm.
"""

import os
import threading
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from darija_rag.answer import Answerer
from darija_rag.factory import Factory, parse_spec
from darija_rag.ingest import Chunk, load_chunks
from darija_rag.llm import LLMError

DEFAULT_RETRIEVER = "dense-m3+rewrite+rerank"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    retriever: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)
    retriever: str | None = None


def _chunk_json(chunk: Chunk, **extra) -> dict:
    return {"id": chunk.id, "doc": chunk.doc, "page": chunk.page, "text": chunk.text, **extra}


def create_app(factory: Factory | None = None, default_retriever: str | None = None) -> FastAPI:
    app = FastAPI(title="darija-rag", version="0.4.0")
    default_retriever = default_retriever or os.environ.get(
        "DARIJA_RAG_RETRIEVER", DEFAULT_RETRIEVER
    )
    state = {"factory": factory}
    lock = threading.Lock()

    def get_factory() -> Factory:
        # Loaded on first use so the app (and its tests) can start without data.
        with lock:
            if state["factory"] is None:
                path = Path(os.environ.get("DARIJA_RAG_CHUNKS", "data/processed/chunks.jsonl"))
                if not path.exists():
                    raise HTTPException(503, f"{path} not found; run darija-rag ingest first")
                state["factory"] = Factory(load_chunks(path), cache_dir=path.parent)
            return state["factory"]

    def spec_of(requested: str | None) -> str:
        spec = requested or default_retriever
        try:
            parse_spec(spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return spec

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("darija_rag").joinpath("web/index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        f = get_factory()
        llm = {"url": f.llm.base_url}
        try:
            llm.update(ok=True, model=f.llm.resolve_model())
        except LLMError as exc:
            llm.update(ok=False, error=str(exc))
        return {"chunks": len(f.chunks), "retriever": default_retriever, "llm": llm}

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict:
        spec = spec_of(req.retriever)
        try:
            results = get_factory().build(spec).retrieve(req.query, req.k)
        except LLMError as exc:  # +rewrite needs the LLM
            raise HTTPException(503, str(exc)) from exc
        return {
            "retriever": spec,
            "results": [_chunk_json(c, score=round(s, 4)) for c, s in results],
        }

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        spec = spec_of(req.retriever)
        f = get_factory()
        rewriter = f.rewriter() if f.uses_rewrite(spec) else None
        try:
            answer = Answerer(f.build(spec), f.llm, k=req.k, rewriter=rewriter).ask(req.question)
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "answer": answer.text,
            "abstained": answer.abstained,
            "variety": answer.variety,
            "retrieval_query": answer.retrieval_query,
            "retriever": spec,
            "model": f.llm.model,
            "citations": [_chunk_json(answer.sources[i], n=i + 1) for i in answer.cited],
            "sources": [_chunk_json(c, n=i) for i, c in enumerate(answer.sources, start=1)],
            "timings": {k: round(v, 3) for k, v in answer.timings.items()},
        }

    return app


app = create_app()
