"""The PDF <-> Markdown contract: first-line header, instructions, and atomic writes.

Every ``foo.pdf`` has a sibling ``foo.md``. The first line records, as an HTML comment so
Obsidian does not render it, the MD5 of the PDF it was generated from plus a hash of the
correction instructions that were active at generation time::

    <!-- ocr-md5: 9f86…015 | instr: 4a7b…c2 -->

The (optional) correction instructions live at the **end** of the file under a visible
``## OCR Instructions`` heading. They are written by the user to nudge the next
recognition pass and are preserved verbatim across regenerations. A change to that section
(detected via the ``instr:`` hash) marks the PDF stale so it is re-recognized.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from .hashing import md5_of_file

# First-line header: a 32-hex MD5 plus an optional instruction hash token.
_HASH_RE = re.compile(r"\b([0-9a-fA-F]{32})\b")
_INSTR_RE = re.compile(r"instr:\s*([0-9a-fA-F]{32}|-)", re.IGNORECASE)
_HEADER_TEMPLATE = "<!-- ocr-md5: {md5} | instr: {instr} -->"

# The instructions section is delimited by a visible heading, matched leniently so the
# user can write any heading level. Canonical form we emit is ``## OCR Instructions``.
INSTRUCTIONS_HEADING = "## OCR Instructions"
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+OCR Instructions[ \t]*$", re.IGNORECASE | re.MULTILINE)

_NO_INSTR = "-"


def md_path_for(pdf_path: Path) -> Path:
    """Return the sibling ``.md`` path for a given PDF path."""
    return pdf_path.with_suffix(".md")


# -- header parsing --------------------------------------------------------------


def read_header(md_path: Path) -> tuple[str | None, str | None]:
    """Return ``(md5, instr_token)`` parsed from the first line, each None if absent."""
    try:
        with open(md_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline()
    except (FileNotFoundError, NotADirectoryError):
        return None, None
    md5_match = _HASH_RE.search(first_line)
    instr_match = _INSTR_RE.search(first_line)
    md5 = md5_match.group(1).lower() if md5_match else None
    instr = instr_match.group(1).lower() if instr_match else None
    return md5, instr


def read_recorded_hash(md_path: Path) -> str | None:
    """Return the MD5 recorded on the first line of ``md_path``, or None."""
    return read_header(md_path)[0]


# -- instructions ----------------------------------------------------------------


def read_instructions(md_path: Path) -> str | None:
    """Return the text of the trailing ``## OCR Instructions`` section, or None.

    Uses the *last* matching heading so an instructions heading the model may have echoed
    inside the body cannot shadow the real (appended) one.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return None
    body = text[matches[-1].end():].strip()
    return body or None


def normalize_instructions(text: str) -> str:
    """Canonicalize instruction text for stable hashing (trim + strip trailing spaces)."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def instructions_hash(text: str | None) -> str:
    """Return a stable token for the instructions: ``-`` when there are none."""
    if not text or not text.strip():
        return _NO_INSTR
    import hashlib

    return hashlib.md5(normalize_instructions(text).encode("utf-8")).hexdigest()


def _strip_instructions_section(body: str) -> str:
    """Remove any trailing instructions heading + content from a model-produced body."""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return body
    return body[: matches[-1].start()].rstrip()


# -- staleness -------------------------------------------------------------------


def is_stale(pdf_path: Path) -> bool:
    """True if the PDF needs (re)processing.

    Stale when the ``.md`` is missing, the first line has no parseable MD5, the recorded
    MD5 differs from the PDF's current bytes, or the recorded instruction hash differs
    from the instructions currently in the file.
    """
    md_path = md_path_for(pdf_path)
    recorded_md5, recorded_instr = read_header(md_path)
    if recorded_md5 is None:
        return True
    if recorded_md5 != md5_of_file(pdf_path):
        return True
    current_instr = instructions_hash(read_instructions(md_path))
    return current_instr != (recorded_instr or _NO_INSTR)


# -- writing ---------------------------------------------------------------------


def build_document(md5: str, body: str, instructions: str | None = None) -> str:
    """Compose the full Markdown document: header, body, then any instructions section."""
    header = _HEADER_TEMPLATE.format(md5=md5, instr=instructions_hash(instructions))
    body = _strip_instructions_section(body)
    document = f"{header}\n\n{body.rstrip()}\n"
    if instructions and instructions.strip():
        document += f"\n{INSTRUCTIONS_HEADING}\n\n{instructions.strip()}\n"
    return document


def write_md(
    pdf_path: Path,
    body: str,
    *,
    pdf_hash: str | None = None,
    instructions: str | None = None,
) -> Path:
    """Atomically write the transcription for ``pdf_path``.

    The MD5 header is computed from the current PDF bytes unless ``pdf_hash`` is supplied.
    ``instructions`` (if any) are appended under the ``## OCR Instructions`` heading and
    their hash is recorded in the header so they survive — and drive — future passes. The
    content is written to a temp file then ``os.replace``-d into place so readers and our
    own watcher never observe a partial file.
    """
    if pdf_hash is None:
        pdf_hash = md5_of_file(pdf_path)
    md_path = md_path_for(pdf_path)
    document = build_document(pdf_hash, body, instructions)

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
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return md_path
