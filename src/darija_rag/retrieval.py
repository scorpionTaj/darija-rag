"""Retrievers share one signature, retrieve(query, k) -> [(chunk, score)], so the eval harness
can compare them and the hybrid retriever can fuse any of them."""

from collections import defaultdict
from typing import Protocol

from darija_rag.bm25 import BM25
from darija_rag.ingest import Chunk
from darija_rag.normalize import tokenize


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25([tokenize(chunk.text) for chunk in chunks])

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        return [(self.chunks[i], score) for i, score in self.index.search(tokenize(query), k)]


class HybridRetriever:
    """Reciprocal rank fusion: rank-based, so BM25 and cosine scores need no calibration."""

    def __init__(self, retrievers: list[Retriever], depth: int = 50, rrf_k: int = 60):
        self.retrievers = retrievers
        self.depth = depth
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        scores: dict[str, float] = defaultdict(float)
        by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, (chunk, _) in enumerate(retriever.retrieve(query, self.depth), start=1):
                scores[chunk.id] += 1 / (self.rrf_k + rank)
                by_id[chunk.id] = chunk
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(by_id[chunk_id], score) for chunk_id, score in ranked]
