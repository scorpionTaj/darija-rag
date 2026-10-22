"""Arabic text normalization and tokenization shared by indexing and querying.

Indexing and querying must use the exact same pipeline, otherwise query terms
silently stop matching document terms.
"""

import re
import unicodedata

# Harakat (fatha, damma, kasra, tanwin, shadda, sukun) and superscript alef.
_DIACRITICS = re.compile(r"[ً-ْٰ]")
_TATWEEL = "ـ"
_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        # Arabic-Indic and Eastern Arabic-Indic digits -> ASCII, so "المادة 12" matches "١٢".
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
    }
)
_TOKEN = re.compile(r"\w+")
# Definite article with optional attached conjunction/preposition: ال، وال، بال، كال، فال، لل.
_ARTICLE = re.compile(r"^(?:[وفبك]?ال|لل)")

# fmt: off
STOPWORDS = frozenset({
    # Arabic (already normalized form)
    "في", "من", "علي", "الي", "عن", "مع", "او", "و", "ان", "هذا", "هذه", "ذلك", "التي", "الذي",
    "الذين", "ما", "لا", "لم", "لن", "قد", "كل", "بين", "به", "بها", "هو", "هي", "كان", "كانت",
    "ثم", "حيث", "اذا", "عند", "غير", "بعد", "قبل",
    # Darija function words
    "ديال", "ديالو", "ديالها", "واش", "شنو", "اش", "كيفاش", "علاش", "فين", "باش",
    # French
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "en", "dans", "au", "aux",
    "par", "pour", "sur", "est", "que", "qui", "ne", "pas", "se", "sa", "son",
})
# fmt: on


def normalize(text: str) -> str:
    # NFKC folds Arabic presentation forms (common in PDF extraction) back to base letters.
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    return text.translate(_LETTER_MAP).lower()


def strip_article(token: str) -> str:
    stripped = _ARTICLE.sub("", token)
    return stripped if len(stripped) >= 2 else token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN.findall(normalize(text)):
        if raw in STOPWORDS:
            continue
        token = strip_article(raw)
        if token not in STOPWORDS:
            tokens.append(token)
    return tokens
