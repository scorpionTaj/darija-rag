"""Build retrievers from spec strings shared by the CLI, the eval harness and the API.

A spec is a base retriever plus optional stages joined by '+', applied left to right:
    bm25 | dense | dense-m3 | hybrid | hybrid-m3      base
    +rewrite                                          Darija/Arabizi -> MSA query rewriting (LLM)
    +rerank                                           cross-encoder reranking of the top 30
Examples: "bm25", "hybrid+rerank", "hybrid+rewrite+rerank".

Heavy components (embedding matrices, models, the LLM client) are built once per Factory and
shared by every spec that needs them.
"""

import threading
from pathlib import Path

from darija_rag.ingest import Chunk
from darija_rag.llm import LLMClient
from darija_rag.retrieval import BM25Retriever, HybridRetriever, Retriever

BASES = ("bm25", "dense", "dense-m3", "hybrid", "hybrid-m3")
STAGES = ("rewrite", "rerank")


def parse_spec(spec: str) -> tuple[str, list[str]]:
    base, *stages = spec.split("+")
    if base not in BASES:
        raise ValueError(f"unknown retriever {base!r}; choose from {', '.join(BASES)}")
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    if len(set(stages)) != len(stages):
        raise ValueError(f"repeated stage in {spec!r}")
    return base, stages


class Factory:
    def __init__(
        self,
        chunks: list[Chunk],
        llm: LLMClient | None = None,
        cache_dir: Path = Path("data/processed"),
    ):
        self.chunks = chunks
        self._llm = llm
        self.cache_dir = cache_dir
        self._built: dict[str, object] = {}
        # Re-entrant: building "hybrid" builds "bm25" inside the same call. Serializing builds
        # also stops concurrent API requests from loading the same model twice.
        self._lock = threading.RLock()

    def _once(self, key: str, make):
        with self._lock:
            if key not in self._built:
                self._built[key] = make()
            return self._built[key]

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient.from_env()
        return self._llm

    def rewriter(self):
        from darija_rag.rewrite import QueryRewriter

        return self._once(
            "rewriter", lambda: QueryRewriter(self.llm, self.cache_dir / "rewrites.json")
        )

    def _dense(self, model_key: str) -> Retriever:
        from darija_rag.dense import MODELS, DenseRetriever

        return self._once(
            f"dense:{model_key}",
            lambda: DenseRetriever(self.chunks, MODELS[model_key], self.cache_dir / "embeddings"),
        )

    def _base(self, base: str) -> Retriever:
        if base == "bm25":
            return self._once("bm25", lambda: BM25Retriever(self.chunks))
        if base == "dense":
            return self._dense("e5")
        if base == "dense-m3":
            return self._dense("m3")
        model_key = "m3" if base == "hybrid-m3" else "e5"
        return self._once(
            base, lambda: HybridRetriever([self._base("bm25"), self._dense(model_key)])
        )

    def build(self, spec: str) -> Retriever:
        base, stages = parse_spec(spec)

        def make() -> Retriever:
            retriever = self._base(base)
            for stage in stages:
                if stage == "rewrite":
                    from darija_rag.rewrite import RewriteRetriever

                    retriever = RewriteRetriever(retriever, self.rewriter())
                else:
                    from darija_rag.rerank import RerankRetriever

                    model = self._once("reranker", _cross_encoder)
                    query_for = self.rewriter().rewrite if "rewrite" in stages else None
                    retriever = RerankRetriever(retriever, model=model, query_for=query_for)
            return retriever

        return self._once(f"spec:{spec}", make)

    def uses_rewrite(self, spec: str) -> bool:
        return "rewrite" in parse_spec(spec)[1]


def _cross_encoder():
    from sentence_transformers import CrossEncoder

    from darija_rag.dense import torch_device
    from darija_rag.rerank import DEFAULT_RERANKER

    return CrossEncoder(DEFAULT_RERANKER, max_length=512, device=torch_device())
