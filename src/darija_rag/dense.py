"""Dense retrieval with a sentence-embedding model, brute-force cosine over a NumPy matrix.

A few thousand chunks fit in memory and search in well under a millisecond, so a vector
database only becomes worth it once the API serves a much larger corpus.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from darija_rag.ingest import Chunk


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""


MODELS = {
    # E5 models are trained with these prefixes; leaving them out costs recall.
    "e5": EmbeddingModel("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "m3": EmbeddingModel("BAAI/bge-m3"),
}


def torch_device() -> str:
    """Device for the embedding and reranking models (DARIJA_RAG_DEVICE: cpu, mps, cuda).

    Defaults to CPU: on a Mac running the LLM locally (LM Studio/MLX), sharing the GPU made
    generation slower and retrieval latency erratic (measured 2-24 s vs a steady 7 s on CPU).
    """
    return os.environ.get("DARIJA_RAG_DEVICE", "cpu")


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model: EmbeddingModel = MODELS["e5"],
        cache_dir: Path | None = Path("data/processed/embeddings"),
        encoder=None,
    ):
        if encoder is None:
            # Imported lazily: torch is an optional extra and slow to import.
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model.name, device=torch_device())
        self.model = model
        self.encoder = encoder
        self.chunks = chunks
        self.embeddings = self._passage_embeddings(cache_dir)

    def _passage_embeddings(self, cache_dir: Path | None) -> np.ndarray:
        digest = hashlib.sha256(f"{self.model}".encode())
        for chunk in self.chunks:
            digest.update(chunk.id.encode() + b"\0" + chunk.text.encode() + b"\0")
        cache = cache_dir / f"{digest.hexdigest()[:16]}.npy" if cache_dir else None
        if cache and cache.exists():
            return np.load(cache)
        embeddings = self._encode([self.model.passage_prefix + chunk.text for chunk in self.chunks])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, embeddings)
        return embeddings

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, batch_size=16, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        scores = self.embeddings @ self._encode([self.model.query_prefix + query])[0]
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
