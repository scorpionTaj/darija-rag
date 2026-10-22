import numpy as np
import pytest

from darija_rag.bm25 import BM25
from darija_rag.dense import DenseRetriever, EmbeddingModel
from darija_rag.evaluate import evaluate
from darija_rag.ingest import Chunk
from darija_rag.retrieval import BM25Retriever, HybridRetriever

CHUNKS = [
    Chunk("labour#p1-c0", "labour.pdf", 1, "مدة العمل الأسبوعية للأجير لا تتجاوز 44 ساعة"),
    Chunk("labour#p2-c0", "labour.pdf", 2, "العطلة السنوية المؤدى عنها يوم ونصف عن كل شهر"),
    Chunk("family#p5-c0", "family.pdf", 5, "الطلاق حل ميثاق الزوجية يمارسه الزوج والزوجة"),
]


def test_bm25_prefers_rarer_and_more_frequent_terms():
    index = BM25([["a", "b"], ["a", "a", "c"], ["d"]])
    ranked = [i for i, _ in index.search(["a", "c"])]
    assert ranked == [1, 0]
    assert index.search(["zzz"]) == []


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25([])


def test_retriever_finds_page_despite_spelling_differences():
    retriever = BM25Retriever(CHUNKS)
    [(top, _)] = retriever.retrieve("العطله السنويه", k=1)
    assert (top.doc, top.page) == ("labour.pdf", 2)


def test_evaluate_reports_recall_and_mrr():
    retriever = BM25Retriever(CHUNKS)
    questions = [
        {"question": "ميثاق الزوجية", "doc": "family.pdf", "page": 5},
        {"question": "ساعات العمل", "doc": "labour.pdf", "page": 1},
        {"question": "شيء غير موجود", "doc": "labour.pdf", "page": 9},
    ]
    for number, q in enumerate(questions, start=1):
        q.update(id=f"line{number}", relevant=[{"doc": q.pop("doc"), "page": q.pop("page")}])
    report = evaluate(retriever.retrieve, questions, ks=(1, 3))
    assert report["recall@1"] == pytest.approx(2 / 3)
    assert report["mrr@3"] == pytest.approx(2 / 3)
    assert report["misses"] == ["line3"]


class _Fixed:
    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, k=10):
        return [(CHUNKS[i], 1.0) for i in self.ranking][:k]


def test_hybrid_rewards_agreement_between_retrievers():
    hybrid = HybridRetriever([_Fixed([0, 1, 2]), _Fixed([1, 2, 0])])
    assert [c.id for c, _ in hybrid.retrieve("q", k=2)] == ["labour#p2-c0", "labour#p1-c0"]


class _FakeEncoder:
    """Bag-of-letters vectors: enough to check prefixes, ranking and caching."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings):
        self.calls.append(texts)
        vectors = np.array([[t.count(ch) for ch in "عطلزوجس"] for t in texts], dtype=float) + 1e-9
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_dense_retriever_uses_e5_prefixes_and_caches(tmp_path):
    encoder = _FakeEncoder()
    dense = DenseRetriever(
        CHUNKS, EmbeddingModel("fake", "query: ", "passage: "), tmp_path, encoder=encoder
    )
    assert all(t.startswith("passage: ") for t in encoder.calls[0])
    [(top, _)] = dense.retrieve("الزوجية", k=1)
    assert top.doc == "family.pdf"
    assert encoder.calls[-1] == ["query: الزوجية"]

    DenseRetriever(
        CHUNKS, EmbeddingModel("fake", "query: ", "passage: "), tmp_path, encoder=encoder
    )
    assert len(encoder.calls) == 2  # passages came from the cache, not re-encoded


def test_load_questions_accepts_both_schemas(tmp_path):
    from darija_rag.evaluate import group, load_questions

    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"question": "a", "doc": "x.pdf", "page": 3, "lang": "msa"}\n'
        '{"id": "q2", "question": "b", "lang": "fr", "relevant": []}\n'
        '{"id": "q3", "question": "c", "lang": "fr", "type": "cross_lingual",'
        ' "relevant": [{"doc": "x.pdf", "page": 1}, {"doc": "y.pdf", "page": 2}]}\n',
        encoding="utf-8",
    )
    first, second, third = load_questions(path)
    assert first["relevant"] == [{"doc": "x.pdf", "page": 3}] and first["type"] == "factual"
    assert second["type"] == "unanswerable"
    assert set(group([first, second, third])) == {
        "all",
        "lang=fr",
        "lang=msa",
        "type=cross_lingual",
        "type=factual",
        "type=unanswerable",
    }


def test_load_questions_rejects_bad_pages(tmp_path):
    from darija_rag.evaluate import load_questions

    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "a", "relevant": [{"doc": "x", "page": true}]}\n', "utf-8")
    with pytest.raises(ValueError):
        load_questions(path)


def test_any_relevant_page_counts():
    from darija_rag.evaluate import first_relevant_rank

    q = {"relevant": [{"doc": "family.pdf", "page": 5}, {"doc": "labour.pdf", "page": 2}]}
    results = [(CHUNKS[0], 1.0), (CHUNKS[1], 0.9), (CHUNKS[2], 0.8)]
    assert first_relevant_rank(q, results) == 2


def test_rerank_reorders_shortlist():
    from darija_rag.rerank import RerankRetriever

    class Scorer:
        def predict(self, pairs, batch_size):
            return [len(text) for _, text in pairs]  # longest chunk wins

    reranked = RerankRetriever(_Fixed([0, 1, 2]), model=Scorer()).retrieve("q", k=2)
    longest = max(CHUNKS, key=lambda c: len(c.text))
    assert reranked[0][0] == longest and len(reranked) == 2


def test_rerank_scores_the_transformed_query():
    from darija_rag.rerank import RerankRetriever

    seen = []

    class Scorer:
        def predict(self, pairs, batch_size):
            seen.extend(q for q, _ in pairs)
            return [0.0] * len(pairs)

    RerankRetriever(_Fixed([0, 1]), model=Scorer(), query_for=lambda q: "MSA:" + q).retrieve("x")
    assert set(seen) == {"MSA:x"}


def test_stratified_sample_covers_every_cell():
    from darija_rag.evaluate import stratified_sample

    questions = [{"id": f"m{i}", "lang": "msa", "type": "factual"} for i in range(90)]
    questions += [{"id": f"a{i}", "lang": "arabizi", "type": "factual"} for i in range(9)]
    questions += [{"id": "u0", "lang": "fr", "type": "unanswerable"}]
    sample = stratified_sample(questions, 20, seed=1)
    ids = [q["id"] for q in sample]
    assert "u0" in ids and sum(i.startswith("a") for i in ids) == 2
    assert 19 <= len(sample) <= 22
    assert ids == sorted(ids, key=[q["id"] for q in questions].index)  # file order kept
    assert stratified_sample(questions, 20, seed=1) == sample  # deterministic
