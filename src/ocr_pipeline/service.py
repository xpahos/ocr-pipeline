"""Wire together reconciliation, watching, and the serial pipeline worker."""

from __future__ import annotations

import threading
from pathlib import Path

import structlog
from watchdog.observers import Observer

from .config import Settings
from .mdfile import is_stale
from .pipeline.graph import Pipeline
from .queue import WorkQueue
from .reconcile import reconcile
from .watcher import DebouncedSubmitter, PdfEventHandler

log = structlog.get_logger(__name__)


class Service:
    """Long-lived service: startup reconciliation followed by live monitoring."""

    def __init__(self, settings: Settings, pipeline: Pipeline | None = None) -> None:
        self.settings = settings
        self.pipeline = pipeline or Pipeline(settings)
        self.queue = WorkQueue(self._handle)
        self._observer: Observer | None = None
        self._stop = threading.Event()

    def _handle(self, pdf_path: Path) -> None:
        """Process one PDF, re-checking staleness against its latest bytes first."""
        if not pdf_path.is_file():
            return
        if not is_stale(pdf_path):
            log.debug("skip_fresh", pdf=str(pdf_path))
            return
        log.info("processing", pdf=str(pdf_path))
        self.pipeline.process(pdf_path)

    def run(self) -> None:
        root = self.settings.vault_root
        if not root.is_dir():
            raise NotADirectoryError(f"OCR_VAULT_ROOT is not a directory: {root}")

        self.queue.start()

        # Live watcher first, so events during the (possibly long) reconcile aren't lost.
        submitter = DebouncedSubmitter(self.settings.debounce_seconds, self.queue.submit)
        handler = PdfEventHandler(root, self.settings.ignore_globs, submitter)
        self._observer = Observer()
        self._observer.schedule(handler, str(root), recursive=True)
        self._observer.start()
        log.info("watching", root=str(root))

        reconcile(root, self.settings.ignore_globs, self.queue.submit)

        try:
            self._stop.wait()
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            self._shutdown(submitter)

    def stop(self) -> None:
        self._stop.set()

    def _shutdown(self, submitter: DebouncedSubmitter) -> None:
        log.info("shutting_down")
        submitter.cancel_all()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        self.queue.stop()
