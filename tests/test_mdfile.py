from __future__ import annotations

from pathlib import Path

from ocr_pipeline.hashing import md5_of_file
from ocr_pipeline.mdfile import (
    build_document,
    is_stale,
    md_path_for,
    read_recorded_hash,
    write_md,
)


def test_build_document_has_hash_on_first_line():
    doc = build_document("0" * 32, "body text")
    first_line = doc.splitlines()[0]
    assert "0" * 32 in first_line
    assert first_line.startswith("<!--")
    assert "body text" in doc


def test_write_md_round_trip(pdf_factory):
    pdf = pdf_factory("note.pdf")
    md = write_md(pdf, "# Heading\n\ncontent")
    assert md == md_path_for(pdf)
    assert read_recorded_hash(md) == md5_of_file(pdf)
    assert "# Heading" in md.read_text(encoding="utf-8")


def test_write_md_is_atomic_no_temp_left(pdf_factory):
    pdf = pdf_factory("note.pdf")
    write_md(pdf, "content")
    leftovers = list(pdf.parent.glob("*.tmp"))
    assert leftovers == []


def test_read_recorded_hash_missing(tmp_path):
    assert read_recorded_hash(tmp_path / "nope.md") is None


def test_read_recorded_hash_malformed_first_line(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("just a normal note\nsecond line", encoding="utf-8")
    assert read_recorded_hash(md) is None


def test_is_stale_missing_md(pdf_factory):
    pdf = pdf_factory("a.pdf")
    assert is_stale(pdf) is True


def test_is_stale_matching_hash(pdf_factory):
    pdf = pdf_factory("a.pdf")
    write_md(pdf, "body")
    assert is_stale(pdf) is False


def test_is_stale_wrong_hash(pdf_factory):
    pdf = pdf_factory("a.pdf")
    md = md_path_for(pdf)
    md.write_text("<!-- ocr-md5: %s -->\n\nold body" % ("f" * 32), encoding="utf-8")
    assert is_stale(pdf) is True


def test_is_stale_after_pdf_changes(pdf_factory, tmp_path):
    pdf = pdf_factory("a.pdf")
    write_md(pdf, "body")
    assert is_stale(pdf) is False
    # Mutate the PDF bytes -> recorded hash no longer matches.
    pdf.write_bytes(pdf.read_bytes() + b"%extra")
    assert is_stale(pdf) is True
