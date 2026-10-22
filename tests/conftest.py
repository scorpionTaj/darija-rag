import pytest

from darija_rag.ingest import Chunk

CHUNKS = [
    Chunk(
        "labour#p1-c0", "labour.pdf", 1, "مدة العمل الأسبوعية للأجير لا تتجاوز 44 ساعة في الأسبوع"
    ),
    Chunk("labour#p2-c0", "labour.pdf", 2, "العطلة السنوية المؤدى عنها يوم ونصف عن كل شهر"),
    Chunk("family#p5-c0", "family.pdf", 5, "الطلاق حل ميثاق الزوجية يمارسه الزوج والزوجة"),
]


class FakeLLM:
    """Stands in for LLMClient: replies come from a function of the last user message."""

    base_url = "http://fake/v1"

    def __init__(self, reply):
        self.reply = reply
        self.model = "fake-model"
        self.calls = []

    def resolve_model(self):
        return self.model

    def chat(self, messages, temperature=0.0, max_tokens=600):
        self.calls.append(messages)
        return self.reply(messages[-1]["content"])


@pytest.fixture
def chunks():
    return list(CHUNKS)
