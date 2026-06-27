"""PDF inspection and splitting to fit OpenAI's per-request limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter


@dataclass(frozen=True)
class PdfInfo:
    page_count: int
    size_bytes: int


def inspect(path: Path) -> PdfInfo:
    """Return page count and byte size; raises if the PDF cannot be opened."""
    reader = PdfReader(str(path))
    return PdfInfo(page_count=len(reader.pages), size_bytes=path.stat().st_size)


def needs_split(info: PdfInfo, *, max_pages: int, max_bytes: int) -> bool:
    return info.page_count > max_pages or info.size_bytes > max_bytes


def _write_range(reader: PdfReader, start: int, end: int, dest: Path) -> int:
    """Write pages [start, end) of ``reader`` to ``dest``; return the byte size."""
    writer = PdfWriter()
    for page_index in range(start, end):
        writer.add_page(reader.pages[page_index])
    with open(dest, "wb") as fh:
        writer.write(fh)
    return dest.stat().st_size


def split_pdf(
    path: Path, out_dir: Path, *, max_pages: int, max_bytes: int
) -> list[Path]:
    """Split ``path`` into ordered chunk PDFs in ``out_dir``, each within both limits.

    Returns ``[path]`` unchanged when the PDF already fits. Splitting is by page range:
    a range that is still too large (by pages or bytes) is bisected until it fits or
    reaches a single page (which is emitted as-is when it cannot be split further).
    """
    info = inspect(path)
    if not needs_split(info, max_pages=max_pages, max_bytes=max_bytes):
        return [path]

    reader = PdfReader(str(path))
    ranges: list[tuple[int, int]] = []

    def recurse(start: int, end: int) -> None:
        pages = end - start
        if pages <= 1:
            ranges.append((start, end))
            return
        if pages <= max_pages:
            # Page count is fine; check byte size by writing a probe.
            probe = out_dir / "_probe.pdf"
            size = _write_range(reader, start, end, probe)
            probe.unlink(missing_ok=True)
            if size <= max_bytes:
                ranges.append((start, end))
                return
        mid = start + pages // 2
        recurse(start, mid)
        recurse(mid, end)

    recurse(0, info.page_count)

    chunks: list[Path] = []
    width = len(str(len(ranges)))
    for idx, (start, end) in enumerate(ranges, start=1):
        dest = out_dir / f"{path.stem}.chunk{idx:0{width}d}.pdf"
        _write_range(reader, start, end, dest)
        chunks.append(dest)
    return chunks
