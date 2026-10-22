"""Cross-encoder reranking: score (query, chunk) pairs jointly over a first-stage shortlist.

Slower than bi-encoders (one forward pass per pair), so it only sees the top `depth` candidates.
"""

from collections.abc import Callable

from darija_rag.dense import torch_device
from darija_rag.ingest import Chunk
from darija_rag.retrieval import Retriever

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class RerankRetriever:
    """query_for maps the user query to the text the cross-encoder scores; after +rewrite that is
    the MSA rewrite, which the cross-encoder understands far better than Arabizi."""

    def __init__(
        self,
        base: Retriever,
        model_name: str = DEFAULT_RERANKER,
        depth: int = 30,
        model=None,
        query_for: Callable[[str], str] | None = None,
    ):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=512, device=torch_device())
        self.base = base
        self.model = model
        self.depth = depth
        self.query_for = query_for or (lambda query: query)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        candidates = [chunk for chunk, _ in self.base.retrieve(query, self.depth)]
        if not candidates:
            return []
        scored_query = self.query_for(query)
        pairs = [(scored_query, chunk.text) for chunk in candidates]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        return [(chunk, float(score)) for chunk, score in ranked[:k]]
