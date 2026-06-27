"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter


def make_pdf(path: Path, pages: int = 1) -> Path:
    """Write a minimal valid PDF with the given number of blank pages."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    def _factory(name: str = "doc.pdf", pages: int = 1) -> Path:
        return make_pdf(tmp_path / name, pages=pages)

    return _factory
