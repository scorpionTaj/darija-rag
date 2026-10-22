import pytest
from conftest import FakeLLM
from fastapi.testclient import TestClient

from darija_rag.answer import NO_ANSWER
from darija_rag.api import create_app
from darija_rag.factory import Factory, parse_spec


def _reply(prompt: str) -> str:
    if "Question:" not in prompt:  # rewrite request
        return "ما هي مدة العمل الأسبوعية للأجير؟"
    return "44 ساعة [1]." if "ساعة" in prompt else NO_ANSWER


@pytest.fixture
def client(chunks, tmp_path):
    factory = Factory(chunks, llm=FakeLLM(_reply), cache_dir=tmp_path)
    return TestClient(create_app(factory, default_retriever="bm25+rewrite"))


def test_parse_spec():
    assert parse_spec("hybrid+rewrite+rerank") == ("hybrid", ["rewrite", "rerank"])
    for bad in ("nope", "bm25+magic", "bm25+rerank+rerank"):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_index_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200 and "Darija RAG" in response.text


def test_health(client):
    body = client.get("/api/health").json()
    assert body["chunks"] == 3 and body["llm"] == {
        "url": "http://fake/v1",
        "ok": True,
        "model": "fake-model",
    }


def test_search(client):
    body = client.post(
        "/api/search", json={"query": "العطلة السنوية", "k": 1, "retriever": "bm25"}
    ).json()
    assert [(r["doc"], r["page"]) for r in body["results"]] == [("labour.pdf", 2)]


def test_ask_darija_is_rewritten_and_cited(client):
    body = client.post("/api/ask", json={"question": "شحال خاصني نخدم فالسيمانة؟"}).json()
    assert body["variety"] == "darija"
    assert body["retrieval_query"] == "ما هي مدة العمل الأسبوعية للأجير؟"
    assert not body["abstained"]
    assert [(c["doc"], c["page"]) for c in body["citations"]] == [("labour.pdf", 1)]


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/ask", json={"question": "q", "retriever": "bad"}).status_code == 422
    assert client.post("/api/ask", json={"question": "q", "k": 50}).status_code == 422


def test_missing_chunks_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DARIJA_RAG_CHUNKS", str(tmp_path / "missing.jsonl"))
    response = TestClient(create_app()).get("/api/health")
    assert response.status_code == 503
