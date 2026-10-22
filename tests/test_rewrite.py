import json

import pytest
from conftest import FakeLLM

from darija_rag.retrieval import BM25Retriever
from darija_rag.rewrite import QueryRewriter, RewriteRetriever, detect_variety


@pytest.mark.parametrize(
    ("text", "variety"),
    [
        ("ما هي مدة الشغل العادية في الأسبوع؟", "msa"),
        ("شحال من ساعة خاصني نخدم فالسيمانة؟", "darija"),
        ("chhal mn sa3a khasni nkhdem f simana?", "arabizi"),
        ("rajli ma b9ach kaysref 3liya", "arabizi"),
        ("Quelle est la fin du contrat le 2e jour ?", "other"),
        ("What is the minimum wage?", "other"),
    ],
)
def test_detect_variety(text, variety):
    assert detect_variety(text) == variety


def test_rewriter_skips_msa_and_caches_darija(tmp_path):
    llm = FakeLLM(lambda _: "ما هي مدة الشغل العادية في الأسبوع؟\nextra line")
    cache = tmp_path / "rewrites.json"
    rewriter = QueryRewriter(llm, cache)

    assert rewriter.rewrite("ما هي مدة الشغل؟") == "ما هي مدة الشغل؟"
    assert llm.calls == []

    darija = "شحال من ساعة خاصني نخدم فالسيمانة؟"
    assert rewriter.rewrite(darija) == "ما هي مدة الشغل العادية في الأسبوع؟"
    rewriter.rewrite(darija)
    assert len(llm.calls) == 1
    # Persisted, so a new process reuses it without calling the LLM.
    assert json.loads(cache.read_text("utf-8"))[darija].startswith("ما هي")
    assert QueryRewriter(FakeLLM(lambda _: "unused"), cache).rewrite(darija).startswith("ما هي")


def test_rewrite_retriever_finds_page_the_darija_query_misses(chunks):
    base = BM25Retriever(chunks)
    darija = "شحال خاصني نخدم فالسيمانة؟"
    assert all(chunk.page != 1 for chunk, _ in base.retrieve(darija, 3))

    rewriter = QueryRewriter(FakeLLM(lambda _: "ما هي مدة العمل الأسبوعية للأجير؟"))
    [(top, _), *_] = RewriteRetriever(base, rewriter).retrieve(darija, 3)
    assert (top.doc, top.page) == ("labour.pdf", 1)
