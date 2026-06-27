"""Recursive vault scan that enforces the PDF <-> MD contract on startup."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Iterator

import structlog

from .mdfile import is_stale, read_recorded_hash

log = structlog.get_logger(__name__)


@dataclass
class ReconcileSummary:
    total: int = 0
    up_to_date: int = 0
    queued: int = 0
    orphans: int = 0


def is_ignored(rel_path: Path, patterns: list[str]) -> bool:
    """True if ``rel_path`` (relative to the vault root) matches an ignore glob."""
    text = rel_path.as_posix()
    for pat in patterns:
        if fnmatch(text, pat):
            return True
        if pat.endswith("/**"):
            base = pat[:-3]
            if text == base or text.startswith(base + "/"):
                return True
    return False


def iter_pdfs(vault_root: Path, ignore_globs: list[str]) -> Iterator[Path]:
    """Yield every non-ignored ``*.pdf`` under ``vault_root`` (recursively)."""
    for pdf in vault_root.rglob("*.pdf"):
        try:
            rel = pdf.relative_to(vault_root)
        except ValueError:
            rel = pdf
        if is_ignored(rel, ignore_globs):
            continue
        yield pdf


def reconcile(
    vault_root: Path,
    ignore_globs: list[str],
    submit: Callable[[Path], None],
) -> ReconcileSummary:
    """Scan the vault, enqueue stale PDFs, and warn about orphan transcriptions."""
    summary = ReconcileSummary()
    for pdf in iter_pdfs(vault_root, ignore_globs):
        summary.total += 1
        if is_stale(pdf):
            summary.queued += 1
            submit(pdf)
        else:
            summary.up_to_date += 1

    summary.orphans = _count_orphans(vault_root, ignore_globs)
    log.info(
        "reconcile_complete",
        total=summary.total,
        up_to_date=summary.up_to_date,
        queued=summary.queued,
        orphans=summary.orphans,
    )
    return summary


def _count_orphans(vault_root: Path, ignore_globs: list[str]) -> int:
    """Warn (do not delete) about transcription ``.md`` files with no sibling PDF.

    Only ``.md`` files that carry our MD5 header are considered transcriptions, so plain
    notes in the vault are never flagged.
    """
    orphans = 0
    for md in vault_root.rglob("*.md"):
        try:
            rel = md.relative_to(vault_root)
        except ValueError:
            rel = md
        if is_ignored(rel, ignore_globs):
            continue
        if read_recorded_hash(md) is None:
            continue  # not one of our transcriptions
        pdf = md.with_suffix(".pdf")
        if not pdf.exists():
            orphans += 1
            log.warning("orphan_md", md=str(md))
    return orphans
