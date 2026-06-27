from __future__ import annotations

import threading
import time
from pathlib import Path

from ocr_pipeline.queue import WorkQueue


def test_queue_processes_submitted(tmp_path):
    seen: list[Path] = []
    q = WorkQueue(seen.append)
    q.start()
    p = tmp_path / "a.pdf"
    p.touch()
    q.submit(p)
    q.join()
    q.stop()
    assert seen == [p.resolve()]


def test_queue_dedupes_pending(tmp_path):
    started = threading.Event()
    release = threading.Event()
    seen: list[Path] = []

    def handler(path: Path) -> None:
        started.set()
        release.wait(timeout=2)
        seen.append(path)

    q = WorkQueue(handler)
    q.start()
    p = tmp_path / "a.pdf"
    p.touch()

    q.submit(p)
    started.wait(timeout=2)  # first job is now running, pending is empty
    # These three collapse to a single queued job while one is in flight.
    q.submit(p)
    q.submit(p)
    q.submit(p)
    release.set()
    q.join()
    q.stop()
    # One in-flight + one coalesced re-queue = 2 (not 4).
    assert len(seen) == 2


def test_queue_serial_execution(tmp_path):
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def handler(path: Path) -> None:
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.02)
        with lock:
            concurrent -= 1

    q = WorkQueue(handler)
    q.start()
    for i in range(5):
        p = tmp_path / f"f{i}.pdf"
        p.touch()
        q.submit(p)
    q.join()
    q.stop()
    assert max_concurrent == 1
