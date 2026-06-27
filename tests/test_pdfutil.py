from __future__ import annotations

from ocr_pipeline.pdfutil import inspect, needs_split, split_pdf


def test_inspect_page_count(pdf_factory):
    pdf = pdf_factory("d.pdf", pages=5)
    info = inspect(pdf)
    assert info.page_count == 5
    assert info.size_bytes > 0


def test_no_split_when_within_limits(pdf_factory, tmp_path):
    pdf = pdf_factory("d.pdf", pages=3)
    out = tmp_path / "out"
    out.mkdir()
    chunks = split_pdf(pdf, out, max_pages=100, max_bytes=33_000_000)
    assert chunks == [pdf]  # original returned unchanged


def test_split_by_page_limit(pdf_factory, tmp_path):
    pdf = pdf_factory("big.pdf", pages=10)
    out = tmp_path / "out"
    out.mkdir()
    chunks = split_pdf(pdf, out, max_pages=3, max_bytes=33_000_000)
    assert len(chunks) > 1
    # Every chunk respects the page limit and they cover all 10 pages.
    total = 0
    for c in chunks:
        info = inspect(c)
        assert info.page_count <= 3
        total += info.page_count
    assert total == 10


def test_split_by_byte_limit(pdf_factory, tmp_path):
    pdf = pdf_factory("big.pdf", pages=8)
    out = tmp_path / "out"
    out.mkdir()
    # Force splitting on size with a tiny byte budget (8 blank pages ~= 1277 bytes).
    chunks = split_pdf(pdf, out, max_pages=100, max_bytes=700)
    assert len(chunks) > 1
    total = sum(inspect(c).page_count for c in chunks)
    assert total == 8


def test_needs_split_flags():
    from ocr_pipeline.pdfutil import PdfInfo

    assert needs_split(PdfInfo(200, 10), max_pages=100, max_bytes=10) is True
    assert needs_split(PdfInfo(5, 99), max_pages=100, max_bytes=10) is True
    assert needs_split(PdfInfo(5, 5), max_pages=100, max_bytes=10) is False
