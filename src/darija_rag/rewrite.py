"""Darija and Arabizi detection, and LLM rewriting into Modern Standard Arabic.

The legal corpus is MSA (and French), so a Darija question like "شحال من ساعة خاصني نخدم
فالسيمانة؟" shares almost no words with the article it asks about. Rewriting it into MSA before
retrieval closes most of that gap. MSA and French questions are passed through untouched, so
they never pay the LLM latency.
"""

import hashlib
import json
import re
import threading
from importlib.resources import files
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


def _load_glossary() -> list[tuple[list[str], list[str], str, bool]]:
    """Rows of (darija forms, arabizi forms, msa, detect) from data/darija_glossary.tsv."""
    rows = []
    text = files("darija_rag").joinpath("data/darija_glossary.tsv").read_text(encoding="utf-8")
    for line in text.splitlines()[2:]:  # skip the comment lines and the header
        if not line.strip() or line.startswith("#"):
            continue
        darija, arabizi, msa, detect = line.split("\t")
        rows.append(
            (darija.split(","), [a.strip() for a in arabizi.split(",")], msa, detect == "1")
        )
    return rows


GLOSSARY = _load_glossary()
# Single-word glossary forms marked detect=1 extend the lexicons; phrases are matched separately.
_DARIJA_WORDS |= {normalize(f) for forms, _, _, d in GLOSSARY if d for f in forms if " " not in f}
# French loanwords used in Darija are hints for rewriting, but not evidence of Arabizi: a French
# question says "contrat" and "salaire" too.
_FRENCH_LOANWORDS = {"contrat", "salaire", "conge", "congé", "patron", "credit", "cnss"}
_ARABIZI_WORDS |= {
    a.lower()
    for _, forms, _, d in GLOSSARY
    if d
    for a in forms
    if " " not in a and a.lower() not in _FRENCH_LOANWORDS
}
_DARIJA_PHRASES = [normalize(f) for forms, _, _, d in GLOSSARY if d for f in forms if " " in f]


def glossary_hints(text: str) -> list[str]:
    """Glossary entries whose Darija or Arabizi form occurs in text, as 'form = MSA' lines."""
    words = re.findall(r"\w+", normalize(text))
    arabic = f" {' '.join(words)} "
    latin = f" {' '.join(w.lower() for w in _LATIN_WORD.findall(text))} "
    hints = []
    for darija, arabizi, msa, _ in GLOSSARY:
        found = [f for f in darija if f" {normalize(f)} " in arabic]
        found += [a for a in arabizi if f" {a.lower()} " in latin]
        if found:
            hints.append(f"{found[0]} = {msa}")
    return hints


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
    normalized = normalize(text)
    words = re.findall(r"\w+", normalized)
    if any(w in _DARIJA_WORDS for w in words) or any(p in normalized for p in _DARIJA_PHRASES):
        return "darija"
    return "msa"


_PROMPT = (
    "You rewrite Moroccan questions about Moroccan law into one formal Modern Standard Arabic "
    "question using the vocabulary of Moroccan legal codes (مدونة الشغل، مدونة الأسرة، قانون "
    "الكراء 67.12، قانون الالتزامات والعقود). "
    "The input may be Darija in Arabic script or Darija in Latin letters (Arabizi, where 3=ع, "
    "7=ح, 9=ق, 2=ء, 5=خ). Keep every fact, number and condition. "
    "Reply with the rewritten question only, no explanation."
)


# Cache entries are only valid for the prompt and glossary that produced them.
_PROMPT_ID = hashlib.sha256((_PROMPT + repr(GLOSSARY)).encode()).hexdigest()[:12]


def build_rewrite_messages(question: str) -> list[dict]:
    hints = glossary_hints(question)
    user = question
    if hints:
        user += "\n\nGlossary (Darija = MSA legal term):\n" + "\n".join(hints)
    return [{"role": "system", "content": _PROMPT}, {"role": "user", "content": user}]


class QueryRewriter:
    def __init__(self, llm: LLMClient, cache_path: Path | None = None):
        self.llm = llm
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if cache_path and cache_path.exists():
            stored = json.loads(cache_path.read_text(encoding="utf-8"))
            if stored.get("prompt_id") == _PROMPT_ID:
                self.cache = stored["rewrites"]

    def rewrite(self, question: str) -> str:
        if detect_variety(question) not in ("darija", "arabizi"):
            return question
        with self._lock:
            if question in self.cache:
                return self.cache[question]
        reply = self.llm.chat(build_rewrite_messages(question), max_tokens=120)
        lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
        rewritten = lines[0] if lines else question
        with self._lock:
            self.cache[question] = rewritten
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".part")
                stored = {"prompt_id": _PROMPT_ID, "rewrites": self.cache}
                tmp.write_text(json.dumps(stored, ensure_ascii=False, indent=1), "utf-8")
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
