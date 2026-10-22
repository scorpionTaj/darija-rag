"""Turn raw PDFs/text files into page-level records, then into overlapping chunks."""

import json
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path

import pymupdf

from darija_rag.ocr import ocr_pdf

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


@dataclass(frozen=True)
class Page:
    doc: str
    page: int  # 1-based, matches what a reader sees in the PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    page: int
    text: str


def read_pages(
    path: Path, ocr_lang: str | None = None, ocr_cache: Path | None = None
) -> Iterator[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf" and ocr_lang:
        texts = ocr_pdf(path, ocr_lang, ocr_cache or path.parent / "ocr")
    elif suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            # Content-stream order: sort=True reorders by position and reverses Arabic words.
            texts = [pdf_page.get_text() for pdf_page in pdf]
    elif suffix == ".txt":
        # Form feeds separate pages, so plain-text exports keep page citations.
        texts = path.read_text(encoding="utf-8").split("\f")
    else:
        raise ValueError(f"unsupported file type: {path}")
    yield from _pages(path.name, texts)


def _pages(doc: str, texts: Iterable[str]) -> Iterator[Page]:
    for number, text in enumerate(texts, start=1):
        if text.strip():
            yield Page(doc, number, _clean(text))


def _clean(text: str) -> str:
    # Keep original spelling for display/citation; only fold presentation forms and whitespace.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def chunk_page(page: Page, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = page.text.split()
    step = size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(
            Chunk(f"{page.doc}#p{page.page}-c{index}", page.doc, page.page, " ".join(window))
        )
        if start + size >= len(words):
            break
    return chunks


def ingest(
    raw_dir: Path,
    out_path: Path,
    size: int = 200,
    overlap: int = 50,
    ocr_patterns: Iterable[str] = (),
    ocr_lang: str = "ara",
) -> int:
    """Chunk every file under raw_dir; PDFs whose name matches an OCR pattern are OCR'd."""
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no .pdf or .txt files found in {raw_dir}")
    ocr_patterns = list(ocr_patterns)
    ocr_cache = out_path.parent / "ocr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            use_ocr = any(fnmatch(path.name, pattern) for pattern in ocr_patterns)
            for page in read_pages(path, ocr_lang if use_ocr else None, ocr_cache):
                for chunk in chunk_page(page, size, overlap):
                    out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                    count += 1
    return count


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
