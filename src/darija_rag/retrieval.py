"""Retrievers share one signature so the eval harness can compare them (BM25 now, dense later)."""

from darija_rag.bm25 import BM25
from darija_rag.ingest import Chunk
from darija_rag.normalize import tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25([tokenize(chunk.text) for chunk in chunks])

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        return [(self.chunks[i], score) for i, score in self.index.search(tokenize(query), k)]
