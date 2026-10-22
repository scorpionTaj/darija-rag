import pymupdf
import pytest

from darija_rag.ingest import Page, chunk_page, ingest, load_chunks, read_pages


def test_chunk_page_overlaps_and_covers_all_words():
    page = Page("a.pdf", 3, " ".join(f"w{i}" for i in range(10)))
    chunks = chunk_page(page, size=4, overlap=1)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert chunks[1].id == "a.pdf#p3-c1"
    assert all(c.page == 3 for c in chunks)


def test_chunk_page_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_page(Page("a", 1, "x"), size=4, overlap=4)


def test_text_files_split_pages_on_form_feed(tmp_path):
    path = tmp_path / "code.txt"
    path.write_text("المادة الأولى\f\fالمادة الثانية", encoding="utf-8")
    pages = list(read_pages(path))
    assert [(p.page, p.text) for p in pages] == [(1, "المادة الأولى"), (3, "المادة الثانية")]


def test_pdf_pages_are_numbered_from_one(tmp_path):
    path = tmp_path / "doc.pdf"
    pdf = pymupdf.open()
    for text in ["first page", "second page"]:
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    assert [(p.page, p.text) for p in read_pages(path)] == [(1, "first page"), (2, "second page")]


def test_ingest_round_trips_through_jsonl(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("مدونة الأسرة", encoding="utf-8")
    out = tmp_path / "chunks.jsonl"
    assert ingest(raw, out) == 1
    [chunk] = load_chunks(out)
    assert (chunk.doc, chunk.page, chunk.text) == ("a.txt", 1, "مدونة الأسرة")


def test_ingest_fails_loudly_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path, tmp_path / "out.jsonl")
