import os

import pymupdf

from darija_rag import ocr
from darija_rag.ingest import ingest, load_chunks


def _pdf(path, pages):
    pdf = pymupdf.open()
    for text in pages:
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_ocr_results_are_cached_per_pdf(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    _pdf(path, ["one", "two"])
    calls = []

    def fake_tesseract(png, lang):
        calls.append(lang)
        return f"صفحة {len(calls)}"

    monkeypatch.setattr(ocr, "_tesseract", fake_tesseract)
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "/usr/bin/tesseract")
    cache = tmp_path / "ocr"
    assert len(ocr.ocr_pdf(path, "ara", cache)) == 2
    assert ocr.ocr_pdf(path, "ara", cache) == ocr.ocr_pdf(path, "ara", cache)
    assert calls == ["ara", "ara"]  # second and third calls hit the cache

    # A newer PDF invalidates the cache.
    later = (cache / "doc.pdf.ara.txt").stat().st_mtime + 10
    os.utime(path, (later, later))
    ocr.ocr_pdf(path, "ara", cache)
    assert len(calls) == 4


def test_ingest_only_ocrs_matching_files(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _pdf(raw / "code_ar.pdf", ["garbled"])
    _pdf(raw / "code_fr.pdf", ["texte propre"])
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr(ocr, "_tesseract", lambda png, lang: "مدونة الشغل")
    out = tmp_path / "processed" / "chunks.jsonl"
    ingest(raw, out, ocr_patterns=["*_ar.pdf"])
    texts = {c.doc: c.text for c in load_chunks(out)}
    assert texts == {"code_ar.pdf": "مدونة الشغل", "code_fr.pdf": "texte propre"}
    assert (out.parent / "ocr" / "code_ar.pdf.ara.txt").exists()
