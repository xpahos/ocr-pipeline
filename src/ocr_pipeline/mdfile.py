"""The PDF <-> Markdown contract: first-line MD5 protocol and atomic writes.

Every ``foo.pdf`` has a sibling ``foo.md`` whose first line records the MD5 of the PDF it
was generated from, stored as an HTML comment so Obsidian does not render it::

    <!-- ocr-md5: 9f86d081884c7d659a2feaa0c55ad015 -->
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from .hashing import md5_of_file

# Matches a first line that contains a 32-hex MD5 token. Tolerant of the exact comment
# wrapper so a hand-written hash line still parses, as long as the hash is on line one.
_HASH_RE = re.compile(r"\b([0-9a-fA-F]{32})\b")

_HEADER_TEMPLATE = "<!-- ocr-md5: {hash} -->"


def md_path_for(pdf_path: Path) -> Path:
    """Return the sibling ``.md`` path for a given PDF path."""
    return pdf_path.with_suffix(".md")


def read_recorded_hash(md_path: Path) -> str | None:
    """Return the MD5 recorded on the first line of ``md_path``, or None.

    Returns None if the file is missing, empty, or the first line has no 32-hex token.
    """
    try:
        with open(md_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline()
    except (FileNotFoundError, NotADirectoryError):
        return None
    match = _HASH_RE.search(first_line)
    return match.group(1).lower() if match else None


def is_stale(pdf_path: Path) -> bool:
    """True if the PDF needs (re)processing.

    A PDF is stale when its ``.md`` is missing, the first line has no parseable hash, or
    the recorded hash differs from the PDF's current MD5.
    """
    recorded = read_recorded_hash(md_path_for(pdf_path))
    if recorded is None:
        return True
    return recorded != md5_of_file(pdf_path)


def build_document(pdf_hash: str, body: str) -> str:
    """Compose the full Markdown document: header line, blank line, then the body."""
    header = _HEADER_TEMPLATE.format(hash=pdf_hash)
    return f"{header}\n\n{body.rstrip()}\n"


def write_md(pdf_path: Path, body: str, *, pdf_hash: str | None = None) -> Path:
    """Atomically write the transcription for ``pdf_path``.

    The MD5 header is computed from the current PDF bytes unless ``pdf_hash`` is supplied
    (callers that already streamed/hashed the file can pass it to avoid a re-read). The
    content is written to a temp file in the same directory then ``os.replace``-d into
    place, so readers and our own watcher never observe a partial file.
    """
    if pdf_hash is None:
        pdf_hash = md5_of_file(pdf_path)
    md_path = md_path_for(pdf_path)
    document = build_document(pdf_hash, body)

    fd, tmp_name = tempfile.mkstemp(
        dir=md_path.parent, prefix=f".{md_path.stem}.", suffix=".md.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(document)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, md_path)
    except BaseException:
        # Best-effort cleanup of the temp file if the rename never happened.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return md_path
