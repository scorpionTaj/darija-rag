"""Okapi BM25 over pre-tokenized chunks, built from scratch on an inverted index."""

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("cannot index an empty corpus")
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.doc_lengths) / len(documents) or 1.0
        # term -> {doc index: term frequency}
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for index, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self.postings[term][index] = freq
        n = len(documents)
        self.idf = {
            term: math.log((n - len(docs) + 0.5) / (len(docs) + 0.5) + 1.0)
            for term, docs in self.postings.items()
        }

    def search(self, query: list[str], k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, freq in self.postings[term].items():
                norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[index] / self.avg_length)
                scores[index] += idf * freq * (self.k1 + 1) / (freq + norm)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
