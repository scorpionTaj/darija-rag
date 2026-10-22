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
    "msa": "لا أجد جوابا لهذا السؤال في النصوص المتوفرة (مدونة الشغل، مدونة الأسرة، قانون "
    "الكراء، قانون الالتزامات والعقود).",
    "darija": "ما لقيتش الجواب ديال هاد السؤال فالنصوص اللي عندي (مدونة الشغل، مدونة الأسرة، "
    "قانون الكرا، قانون الالتزامات والعقود).",
    "arabizi": "Ma l9itch jawab dyal had so2al f nossos li 3ndi (cheghl, l2osra, lkra, "
    "l2iltizamat w l3o9oud).",
    "other": "Je ne trouve pas la réponse dans les textes disponibles (Code du travail, "
    "Code de la famille, loi 67-12 sur les baux, Code des obligations et des contrats).",
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
    language_fixed: bool = False  # the first draft was in the wrong language and was rewritten

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


# Questions whose answers must come back in the same variety. MSA questions accept any Arabic
# answer, and Darija words in an MSA question's answer are harmless.
_CHECKED = {"darija", "arabizi", "other"}

_FIX = (
    "Rewrite the answer below so it is entirely in this language: {rule} Keep every fact, number, "
    "article number and every citation marker like [1] exactly. Reply with the rewritten answer "
    "only."
)


def language_ok(answer: str, variety: str) -> bool:
    if variety not in _CHECKED:
        return True
    return detect_variety(re.sub(r"\[\d+\]", " ", answer)) == variety


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
        generated = time.perf_counter()
        timings = {"retrieve_s": retrieved - start, "generate_s": generated - retrieved}
        rewritten = self.rewriter.rewrite(question) if self.rewriter else question
        if NO_ANSWER in reply or not reply:
            timings["total_s"] = generated - start
            return Answer(
                question, ABSTENTION[variety], True, variety, sources, [], rewritten, timings
            )
        cited = parse_citations(reply, len(sources))
        fixed = False
        if not language_ok(reply, variety):
            reply, fixed = self._fix_language(reply, variety, cited, len(sources))
            timings["fix_s"] = time.perf_counter() - generated
        timings["total_s"] = time.perf_counter() - start
        return Answer(question, reply, False, variety, sources, cited, rewritten, timings, fixed)

    def _fix_language(
        self, reply: str, variety: str, cited: list[int], n_sources: int
    ) -> tuple[str, bool]:
        """One rewrite into the question's language. Kept only if it is in the right language
        and cites exactly the same sources; otherwise the original draft wins."""
        prompt = _FIX.format(rule=_LANGUAGE_RULE[variety])
        candidate = _strip_reasoning(
            self.llm.chat(
                [{"role": "system", "content": prompt}, {"role": "user", "content": reply}],
                max_tokens=600,
            )
        )
        same_citations = sorted(parse_citations(candidate, n_sources)) == sorted(cited)
        if candidate and same_citations and language_ok(candidate, variety):
            return candidate, True
        return reply, False
