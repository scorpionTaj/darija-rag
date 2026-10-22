"""Grounded answer generation: retrieve, prompt the LLM with numbered sources, parse citations.

The model must answer only from the sources and cite them as [n]; when the sources do not
contain the answer it replies with a sentinel, which becomes an explicit abstention instead of
a guessed legal answer.
"""

import re
import time
from dataclasses import dataclass, field

from darija_rag.ingest import Chunk
from darija_rag.llm import LLMClient
from darija_rag.retrieval import Retriever
from darija_rag.rewrite import QueryRewriter, detect_variety

NO_ANSWER = "NO_ANSWER"

_LANGUAGE_RULE = {
    "msa": "Answer in Modern Standard Arabic.",
    "darija": "Answer in simple Moroccan Darija written in Arabic script.",
    "arabizi": "Answer in Moroccan Darija written in Latin letters (Arabizi), like the question.",
    "other": "Answer in the language of the question.",
}

ABSTENTION = {
    "msa": "لا أجد جوابا لهذا السؤال في النصوص المتوفرة (مدونة الشغل ومدونة الأسرة).",
    "darija": "ما لقيتش الجواب ديال هاد السؤال فالنصوص اللي عندي (مدونة الشغل ومدونة الأسرة).",
    "arabizi": "Ma l9itch jawab dyal had so2al f nossos li 3ndi (mdawanat cheghl w l2osra).",
    "other": "Je ne trouve pas la réponse dans les textes disponibles (Code du travail, "
    "Code de la famille).",
}

_SYSTEM = (
    "You are a careful assistant for Moroccan law. Answer ONLY with facts stated in the numbered "
    "sources. After each fact, cite its source like [1] or [2][3]. Mention article numbers when "
    "the source gives them. Be concise: at most 4 sentences. If the sources do not contain the "
    f"answer, reply with exactly {NO_ANSWER} and nothing else. Never use outside knowledge."
)


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    variety: str
    sources: list[Chunk]
    cited: list[int] = field(default_factory=list)  # indexes into sources
    retrieval_query: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def cited_chunks(self) -> list[Chunk]:
        return [self.sources[i] for i in self.cited]


def build_messages(question: str, sources: list[Chunk], variety: str) -> list[dict]:
    blocks = [f"[{i}] ({c.doc}, page {c.page})\n{c.text}" for i, c in enumerate(sources, start=1)]
    # The language rule is repeated after the question: models follow the last instruction best,
    # and otherwise drift into the (MSA) language of the sources.
    user = (
        "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\n"
        f"{_LANGUAGE_RULE[variety]} Cite sources as [n]."
    )
    system = f"{_SYSTEM} {_LANGUAGE_RULE[variety]}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_citations(text: str, n_sources: int) -> list[int]:
    seen = []
    for match in re.finditer(r"\[(\d+)\]", text):
        index = int(match.group(1)) - 1
        if 0 <= index < n_sources and index not in seen:
            seen.append(index)
    return seen


def _strip_reasoning(text: str) -> str:
    # Reasoning models served by LM Studio may prepend <think>...</think>.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        k: int = 5,
        rewriter: QueryRewriter | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.rewriter = rewriter  # only to report the rewritten query; retrieval already used it

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        variety = detect_variety(question)
        start = time.perf_counter()
        sources = [chunk for chunk, _ in self.retriever.retrieve(question, self.k)]
        retrieved = time.perf_counter()
        if not sources:
            reply = NO_ANSWER
        else:
            reply = _strip_reasoning(self.llm.chat(build_messages(question, sources, variety)))
        done = time.perf_counter()
        timings = {
            "retrieve_s": retrieved - start,
            "generate_s": done - retrieved,
            "total_s": done - start,
        }
        rewritten = self.rewriter.rewrite(question) if self.rewriter else question
        cited = parse_citations(reply, len(sources))
        if NO_ANSWER in reply or not reply:
            return Answer(
                question, ABSTENTION[variety], True, variety, sources, [], rewritten, timings
            )
        return Answer(question, reply, False, variety, sources, cited, rewritten, timings)
