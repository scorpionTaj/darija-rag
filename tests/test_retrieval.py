import pytest

from darija_rag.bm25 import BM25
from darija_rag.evaluate import evaluate
from darija_rag.ingest import Chunk
from darija_rag.retrieval import BM25Retriever

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
    report = evaluate(retriever.retrieve, questions, ks=(1, 3))
    assert report["recall@1"] == pytest.approx(2 / 3)
    assert report["mrr@3"] == pytest.approx(2 / 3)
    assert report["misses"] == ["شيء غير موجود"]
