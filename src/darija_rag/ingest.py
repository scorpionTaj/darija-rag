"""Turn raw PDFs/text files into page-level records, then into overlapping chunks."""

import json
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

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


def read_pages(path: Path) -> Iterator[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            for number, pdf_page in enumerate(pdf, start=1):
                text = pdf_page.get_text(sort=True)
                if text.strip():
                    yield Page(path.name, number, _clean(text))
    elif suffix == ".txt":
        # Form feeds separate pages, so plain-text exports keep page citations.
        raw = path.read_text(encoding="utf-8")
        for number, text in enumerate(raw.split("\f"), start=1):
            if text.strip():
                yield Page(path.name, number, _clean(text))
    else:
        raise ValueError(f"unsupported file type: {path}")


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


def ingest(raw_dir: Path, out_path: Path, size: int = 200, overlap: int = 50) -> int:
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no .pdf or .txt files found in {raw_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            for page in read_pages(path):
                for chunk in chunk_page(page, size, overlap):
                    out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                    count += 1
    return count


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
