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


def _slug(model: EmbeddingModel) -> str:
    # The prefixes are part of the key: the same model with other prefixes gives other vectors.
    digest = hashlib.sha256(repr(model).encode()).hexdigest()[:8]
    return f"{model.name.replace('/', '__')}-{digest}"


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
        """Embeddings for every chunk, cached per passage text: adding a document embeds only
        its new chunks instead of the whole corpus."""
        texts = [self.model.passage_prefix + chunk.text for chunk in self.chunks]
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        store = cache_dir / f"{_slug(self.model)}.npz" if cache_dir else None
        cached: dict[str, np.ndarray] = {}
        if store and store.exists():
            with np.load(store) as data:
                cached = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
        missing = sorted({i for i, key in enumerate(keys) if key not in cached})
        if missing:
            fresh = self._encode([texts[i] for i in missing])
            cached.update({keys[i]: vector for i, vector in zip(missing, fresh, strict=True)})
            if store:
                store.parent.mkdir(parents=True, exist_ok=True)
                tmp = store.with_suffix(".part.npz")
                np.savez(tmp, keys=np.array(list(cached)), vectors=np.stack(list(cached.values())))
                tmp.replace(store)
        return np.stack([cached[key] for key in keys]).astype(np.float32)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, batch_size=16, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        scores = self.embeddings @ self._encode([self.model.query_prefix + query])[0]
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
