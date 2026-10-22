"""Darija and Arabizi detection, and LLM rewriting into Modern Standard Arabic.

The legal corpus is MSA (and French), so a Darija question like "شحال من ساعة خاصني نخدم
فالسيمانة؟" shares almost no words with the article it asks about. Rewriting it into MSA before
retrieval closes most of that gap. MSA and French questions are passed through untouched, so
they never pay the LLM latency.
"""

import json
import re
import threading
from pathlib import Path

from darija_rag.ingest import Chunk
from darija_rag.llm import LLMClient
from darija_rag.normalize import normalize
from darija_rag.retrieval import HybridRetriever, Retriever

# Frequent Darija function words and verb forms that MSA legal text never uses. Left out on
# purpose: words shared with MSA (إلا، ولا، غير، بلا، الخدمة).
# fmt: off
_DARIJA_WORDS = frozenset(normalize(w) for w in [
    "شنو", "شنهو", "اشنو", "واش", "كيفاش", "علاش", "فين", "فاين", "شحال", "قداش", "ديال",
    "ديالي", "ديالو", "ديالها", "ديالنا", "ديالكم", "دابا", "بزاف", "مزيان", "خاصني", "خاصو",
    "خاصها", "خاصهم", "خاصنا", "بغيت", "بغا", "كنخدم", "كيخدم", "كنقدر", "نقدر", "كيعطيوني",
    "جراو", "جراني", "الباطرون", "السيمانة", "الكونجي", "الكونطرا", "خدام", "خدامة",
    "الدراري", "راجلي", "مراتي", "غادي", "هادشي", "هاداك", "فالخدمة", "فالسيمانة", "كتاخد",
    "كيتحسب", "ملي", "خاص", "شكون", "اللي", "امتى", "لامتى", "فشحال", "كيولي", "كيتسمى",
    "ولادو",
])
# Not "fin": it is also French.
_ARABIZI_WORDS = frozenset([
    "wach", "chno", "chnou", "kifach", "3lach", "chhal", "dyal", "dyali", "khasni", "khaso",
    "bghit", "daba", "bzaf", "ila", "mli", "rajli", "mrati", "lkhedma", "lkhdma", "nkhdem",
    "ghadi", "bla",
])
# fmt: on
# Words with Arabizi digits for ع ح ق ء خ, like "3am", "7a9", "s9sa".
_ARABIZI_DIGITS = re.compile(r"[a-z]+[23579][a-z]*|[23579][a-z]+", re.IGNORECASE)
_LATIN_WORD = re.compile(r"[A-Za-z0-9']+")
_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")


def detect_variety(text: str) -> str:
    """Return 'arabizi', 'darija', 'msa' or 'other' (French, English...)."""
    if not _ARABIC_LETTER.search(text):
        words = [w.lower() for w in _LATIN_WORD.findall(text)]
        lexicon = sum(w in _ARABIZI_WORDS for w in words)
        digits = sum(bool(_ARABIZI_DIGITS.fullmatch(w)) for w in words)
        # One digit-word alone is weak evidence (French "2e"), a Darija word is strong.
        return "arabizi" if lexicon >= 1 or digits >= 2 else "other"
    words = re.findall(r"\w+", normalize(text))
    return "darija" if any(w in _DARIJA_WORDS for w in words) else "msa"


_PROMPT = (
    "You rewrite Moroccan questions about Moroccan law into one formal Modern Standard Arabic "
    "question using the vocabulary of Moroccan legal codes (مدونة الشغل، مدونة الأسرة). "
    "The input may be Darija in Arabic script or Darija in Latin letters (Arabizi, where 3=ع, "
    "7=ح, 9=ق, 2=ء, 5=خ). Keep every fact, number and condition. "
    "Reply with the rewritten question only, no explanation."
)


class QueryRewriter:
    def __init__(self, llm: LLMClient, cache_path: Path | None = None):
        self.llm = llm
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def rewrite(self, question: str) -> str:
        if detect_variety(question) not in ("darija", "arabizi"):
            return question
        with self._lock:
            if question in self.cache:
                return self.cache[question]
        reply = self.llm.chat(
            [{"role": "system", "content": _PROMPT}, {"role": "user", "content": question}],
            max_tokens=120,
        )
        lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
        rewritten = lines[0] if lines else question
        with self._lock:
            self.cache[question] = rewritten
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".part")
                tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), "utf-8")
                tmp.replace(self.cache_path)
        return rewritten


class RewriteRetriever:
    """Retrieve with both the original and the rewritten query, fused by reciprocal rank.

    Keeping the original guards against a bad rewrite; the fusion lets either query win.
    """

    def __init__(self, base: Retriever, rewriter: QueryRewriter):
        self.base = base
        self.rewriter = rewriter

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        rewritten = self.rewriter.rewrite(query)
        if rewritten == query:
            return self.base.retrieve(query, k)
        both = HybridRetriever([_Fixed(self.base, query), _Fixed(self.base, rewritten)])
        return both.retrieve(query, k)


class _Fixed:
    """Adapter that ignores the incoming query and retrieves for a fixed one."""

    def __init__(self, base: Retriever, query: str):
        self.base = base
        self.query = query

    def retrieve(self, _query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        return self.base.retrieve(self.query, k)
