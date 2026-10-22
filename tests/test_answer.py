import httpx
import pytest
from conftest import FakeLLM

from darija_rag.answer import ABSTENTION, NO_ANSWER, Answerer, parse_citations
from darija_rag.judge import parse_verdict, score, summarize
from darija_rag.llm import LLMClient, LLMError
from darija_rag.retrieval import BM25Retriever


def test_parse_citations_dedupes_and_drops_out_of_range():
    assert parse_citations("a [2] b [1][2] c [9] [0]", n_sources=3) == [1, 0]


def test_answer_with_citations(chunks):
    llm = FakeLLM(lambda _: "<think>hmm</think>44 ساعة في الأسبوع [1].")
    answer = Answerer(BM25Retriever(chunks), llm, k=2).ask("ما هي مدة العمل الأسبوعية؟")
    assert not answer.abstained
    assert answer.text == "44 ساعة في الأسبوع [1]."
    assert [(c.doc, c.page) for c in answer.cited_chunks] == [("labour.pdf", 1)]
    assert answer.variety == "msa"
    prompt = llm.calls[0][-1]["content"]
    assert "[1] (labour.pdf, page 1)" in prompt


def test_abstention_is_localized(chunks):
    llm = FakeLLM(lambda _: NO_ANSWER)
    answer = Answerer(BM25Retriever(chunks), llm).ask("شحال كتخلص ديال الضريبة على الدخل؟ ساعة")
    assert answer.abstained and answer.text == ABSTENTION["darija"]
    assert answer.cited == []


def test_no_sources_abstains_without_calling_llm(chunks):
    llm = FakeLLM(lambda _: "should not be called")
    answer = Answerer(BM25Retriever(chunks), llm).ask("Quel est le taux de TVA ?")
    assert answer.abstained and llm.calls == []
    assert answer.text == ABSTENTION["other"]


def test_empty_question_rejected(chunks):
    with pytest.raises(ValueError):
        Answerer(BM25Retriever(chunks), FakeLLM(str)).ask("   ")


def test_parse_verdict_tolerates_noise():
    assert parse_verdict('Sure: {"correct": true, "faithful": false} done') == {
        "correct": True,
        "faithful": False,
    }
    assert parse_verdict("no json") == {"correct": False, "faithful": False}


def test_score_and_summarize(chunks):
    llm = FakeLLM(lambda _: "44 ساعة [1]")
    answer = Answerer(BM25Retriever(chunks), llm, k=1).ask("مدة العمل الأسبوعية")
    in_scope = {
        "id": "a",
        "lang": "msa",
        "question": "q",
        "answer": "44",
        "relevant": [{"doc": "labour.pdf", "page": 1}],
    }
    out_scope = {"id": "b", "lang": "fr", "question": "q", "answer": "-", "relevant": []}
    rows = [
        score(in_scope, answer, {"correct": True, "faithful": True}),
        score(out_scope, answer, None),  # answered an out-of-scope question: wrong
    ]
    assert rows[0]["citation_hit"] and rows[0]["abstention_ok"]
    assert not rows[1]["correct"] and not rows[1]["abstention_ok"]
    s = summarize(rows)
    assert s["accuracy"] == 0.5 and s["abstention_recall"] == 0.0 and s["false_abstention"] == 0.0


def _client(handler) -> LLMClient:
    client = LLMClient(base_url="http://llm/v1")
    client._http = httpx.Client(base_url="http://llm/v1", transport=httpx.MockTransport(handler))
    return client


def test_llm_client_picks_first_chat_model_and_chats():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "text-embedding-x"}, {"id": "qwen"}]})
        assert b'"model":"qwen"' in request.content.replace(b" ", b"")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert _client(handler).chat([{"role": "user", "content": "hi"}]) == "ok"


def test_llm_client_reports_unreachable_endpoint():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="unreachable"):
        _client(handler).chat([{"role": "user", "content": "hi"}])


def test_llm_client_reports_http_errors():
    client = _client(lambda r: httpx.Response(500, text="boom"))
    client.model = "m"
    with pytest.raises(LLMError, match="500"):
        client.chat([{"role": "user", "content": "hi"}])


def test_llm_client_prefers_model_loaded_in_lm_studio():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "tts-stub", "type": "llm", "state": "loaded", "arch": None},
                        {"id": "big", "type": "vlm", "state": "not-loaded", "arch": "qwen"},
                        {"id": "gemma", "type": "vlm", "state": "loaded", "arch": "gemma"},
                    ]
                },
            )
        raise AssertionError(request.url)

    assert _client(handler).resolve_model() == "gemma"


def test_judge_sees_only_cited_sources(chunks):
    from darija_rag.judge import judge_answer

    class AllChunks:
        def retrieve(self, query, k=10):
            return [(c, 1.0) for c in chunks][:k]

    answer = Answerer(AllChunks(), FakeLLM(lambda _: "44 [2]"), k=3).ask("مدة العمل")
    judge = FakeLLM(lambda _: '{"correct": true, "faithful": true}')
    judge_answer(judge, {"question": "q", "answer": "44"}, answer)
    prompt = judge.calls[0][-1]["content"]
    assert "[2]" in prompt and "[1]" not in prompt and "[3]" not in prompt


def test_answer_table_renders_groups():
    from darija_rag.judge import answer_table

    row = {
        "id": "a",
        "lang": "msa",
        "type": "factual",
        "in_scope": True,
        "abstained": False,
        "correct": True,
        "faithful": True,
        "citation_hit": True,
        "total_s": 3.0,
    }
    table = answer_table([row])
    assert "| all | 1 | 1.00 |" in table and "| lang=msa | 1 |" in table


class _AllChunks:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, k=10):
        return [(c, 1.0) for c in self.chunks][:k]


def _scripted(*replies):
    replies = iter(replies)
    return FakeLLM(lambda _: next(replies))


def test_wrong_language_answer_is_rewritten(chunks):
    llm = _scripted("مدة الشغل هي 44 ساعة في الأسبوع [1].", "إيه، خاصك تخدم 44 ساعة فالسيمانة [1].")
    answer = Answerer(_AllChunks(chunks), llm).ask("شحال من ساعة خاصني نخدم فالسيمانة؟")
    assert answer.language_fixed and answer.text.startswith("إيه")
    assert [c.page for c in answer.cited_chunks] == [1]
    assert len(llm.calls) == 2 and "fix_s" in answer.timings


def test_fix_that_drops_a_citation_is_rejected(chunks):
    llm = _scripted("44 heures [1] et 24 heures [2].", "44 heures [1].")
    answer = Answerer(_AllChunks(chunks), llm).ask("chhal mn sa3a khasni nkhdem f simana?")
    assert not answer.language_fixed
    assert answer.text == "44 heures [1] et 24 heures [2]."


def test_answers_in_the_right_language_skip_the_fix(chunks):
    llm = _scripted("Le salarié travaille 44 heures par semaine [1].")
    answer = Answerer(_AllChunks(chunks), llm).ask("Combien d'heures par semaine ?")
    assert not answer.language_fixed and len(llm.calls) == 1


def test_language_ok():
    from darija_rag.answer import language_ok

    assert language_ok("44 sa3a f simana [1]", "arabizi")
    assert not language_ok("مدة الشغل 44 ساعة [1]", "arabizi")
    assert language_ok("أي جواب", "msa")
