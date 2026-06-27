"""A serial, deduplicated work queue drained by a single background worker thread."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable

import structlog

log = structlog.get_logger(__name__)

Handler = Callable[[Path], None]


class WorkQueue:
    """Single-consumer queue: at most one ``handler`` call runs at a time.

    Paths already *waiting* in the queue are deduplicated, so a burst of events for the
    same file collapses to one job. A path that arrives again while its previous job is
    already running is allowed back in, so the latest bytes are always (re)evaluated.
    """

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._q: "queue.Queue[Path]" = queue.Queue()
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def submit(self, path: Path) -> None:
        path = path.resolve()
        with self._lock:
            if path in self._pending:
                return
            self._pending.add(path)
        self._q.put(path)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="ocr-worker", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            # Drop from pending before handling so a concurrent change re-queues.
            with self._lock:
                self._pending.discard(path)
            try:
                self._handler(path)
            except Exception:
                log.exception("job_failed", path=str(path))
            finally:
                self._q.task_done()

    def join(self) -> None:
        """Block until every submitted job has been processed."""
        self._q.join()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
