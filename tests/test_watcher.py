from __future__ import annotations

from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent

from ocr_pipeline.watcher import PdfEventHandler


class RecordingSubmitter:
    def __init__(self) -> None:
        self.triggered: list[Path] = []

    def trigger(self, path: Path) -> None:
        self.triggered.append(path)


def _handler(tmp_path) -> tuple[PdfEventHandler, RecordingSubmitter]:
    sub = RecordingSubmitter()
    return PdfEventHandler(tmp_path, [".obsidian/**"], sub), sub


def test_pdf_event_triggers_pdf(tmp_path):
    handler, sub = _handler(tmp_path)
    pdf = tmp_path / "a.pdf"
    pdf.touch()
    handler.on_created(FileCreatedEvent(str(pdf)))
    assert sub.triggered == [pdf]


def test_md_event_triggers_sibling_pdf(tmp_path):
    handler, sub = _handler(tmp_path)
    pdf = tmp_path / "a.pdf"
    pdf.touch()
    md = tmp_path / "a.md"
    md.touch()
    handler.on_modified(FileModifiedEvent(str(md)))
    assert sub.triggered == [pdf]  # mapped to the sibling PDF


def test_md_event_without_pdf_is_ignored(tmp_path):
    handler, sub = _handler(tmp_path)
    md = tmp_path / "orphan.md"
    md.touch()
    handler.on_modified(FileModifiedEvent(str(md)))
    assert sub.triggered == []


def test_ignored_path_is_skipped(tmp_path):
    handler, sub = _handler(tmp_path)
    (tmp_path / ".obsidian").mkdir()
    pdf = tmp_path / ".obsidian" / "a.pdf"
    pdf.touch()
    handler.on_created(FileCreatedEvent(str(pdf)))
    assert sub.triggered == []


def test_non_pdf_non_md_ignored(tmp_path):
    handler, sub = _handler(tmp_path)
    other = tmp_path / "a.txt"
    other.touch()
    handler.on_created(FileCreatedEvent(str(other)))
    assert sub.triggered == []
