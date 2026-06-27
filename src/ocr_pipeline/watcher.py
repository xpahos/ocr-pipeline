"""Cross-platform filesystem watcher that feeds changed PDFs into the work queue."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from .reconcile import is_ignored

log = structlog.get_logger(__name__)


class DebouncedSubmitter:
    """Coalesces rapid events for the same path into a single delayed submit.

    Obsidian sync and atomic saves emit bursts of create/modify events; waiting for a
    quiet period avoids enqueueing a file mid-write.
    """

    def __init__(self, delay: float, submit: Callable[[Path], None]) -> None:
        self._delay = delay
        self._submit = submit
        self._timers: dict[Path, threading.Timer] = {}
        self._lock = threading.Lock()

    def trigger(self, path: Path) -> None:
        with self._lock:
            existing = self._timers.get(path)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self._delay, self._fire, args=(path,))
            timer.daemon = True
            self._timers[path] = timer
            timer.start()

    def _fire(self, path: Path) -> None:
        with self._lock:
            self._timers.pop(path, None)
        self._submit(path)

    def cancel_all(self) -> None:
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class PdfEventHandler(FileSystemEventHandler):
    """Routes PDF and sibling-MD events to a debounced submitter (keyed by the PDF).

    A ``.md`` event maps to its sibling ``.pdf`` so that adding/editing the
    ``## OCR Instructions`` section triggers re-recognition. Our own atomic ``.md`` writes
    do not loop: after writing, the recorded hashes match the file, so the requeued PDF is
    immediately "fresh" and skipped by the worker's staleness re-check.
    """

    def __init__(
        self,
        vault_root: Path,
        ignore_globs: list[str],
        submitter: DebouncedSubmitter,
    ) -> None:
        self._vault_root = vault_root
        self._ignore_globs = ignore_globs
        self._submitter = submitter

    def _maybe(self, raw_path: str | bytes) -> None:
        path = Path(os.fsdecode(raw_path)) if isinstance(raw_path, bytes) else Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pdf = path
        elif suffix == ".md":
            pdf = path.with_suffix(".pdf")
            if not pdf.exists():
                return  # plain note, not a transcription for any PDF we know about
        else:
            return
        try:
            rel = path.relative_to(self._vault_root)
        except ValueError:
            rel = path
        if is_ignored(rel, self._ignore_globs):
            return
        log.debug("fs_event", path=str(path), pdf=str(pdf))
        self._submitter.trigger(pdf)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        # A move/rename lands the file at its destination — treat as a new arrival.
        if not event.is_directory:
            self._maybe(event.dest_path)
