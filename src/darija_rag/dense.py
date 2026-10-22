"""Dense retrieval with an E5 sentence-embedding model, brute-force cosine over a NumPy matrix.

A few thousand chunks fit in memory and search in well under a millisecond, so a vector
database only becomes worth it once the API serves a much larger corpus.
"""

import hashlib
from pathlib import Path

import numpy as np

from darija_rag.ingest import Chunk

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = Path("data/processed/embeddings"),
        model=None,
    ):
        if model is None:
            # Imported lazily: torch is an optional extra and slow to import.
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self.model = model
        self.chunks = chunks
        self.embeddings = self._passage_embeddings(model_name, cache_dir)

    def _passage_embeddings(self, model_name: str, cache_dir: Path | None) -> np.ndarray:
        digest = hashlib.sha256(model_name.encode())
        for chunk in self.chunks:
            digest.update(chunk.id.encode() + b"\0" + chunk.text.encode() + b"\0")
        cache = cache_dir / f"{digest.hexdigest()[:16]}.npy" if cache_dir else None
        if cache and cache.exists():
            return np.load(cache)
        # E5 models are trained with these prefixes; leaving them out costs recall.
        embeddings = self._encode([f"passage: {chunk.text}" for chunk in self.chunks])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, embeddings)
        return embeddings

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        scores = self.embeddings @ self._encode([f"query: {query}"])[0]
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
