"""OCR for PDFs whose embedded text layer is unusable (common for Arabic: reversed words,
swapped lam-alef ligatures, scrambled letters).

Calls the tesseract CLI directly: PyMuPDF's built-in OCR text page reverses some
right-to-left lines. Results are cached as one form-feed separated text file per PDF,
because a full OCR pass takes minutes and chunking experiments should not repeat it.
"""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf


def ocr_pdf(path: Path, lang: str, cache_dir: Path, dpi: int = 300) -> list[str]:
    cache = cache_dir / f"{path.name}.{lang}.txt"
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return cache.read_text(encoding="utf-8").split("\f")
    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract is not installed (brew install tesseract tesseract-lang)")

    with pymupdf.open(path) as pdf:
        images = [page.get_pixmap(dpi=dpi).tobytes("png") for page in pdf]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(lambda image: _tesseract(image, lang), images))

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text("\f".join(texts), encoding="utf-8")
    tmp.replace(cache)
    return texts


def _tesseract(png: bytes, lang: str) -> str:
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", lang],
        input=png,
        capture_output=True,
        check=True,
        # One thread per process; parallelism comes from the pool.
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    return result.stdout.decode("utf-8")
