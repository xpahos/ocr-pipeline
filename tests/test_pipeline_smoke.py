from __future__ import annotations

from pathlib import Path

from ocr_pipeline.config import Settings
from ocr_pipeline.hashing import md5_of_file
from ocr_pipeline.mdfile import read_recorded_hash
from ocr_pipeline.pipeline.graph import Pipeline
from ocr_pipeline.pipeline.nodes import CHUNK_SEPARATOR


class FakeRecognizer:
    """Returns deterministic markdown per chunk; records how many times it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, pdf_path: Path, on_delta=None) -> str:
        self.calls += 1
        return f"# Transcription {self.calls}\n\nbody"


def _settings(**overrides) -> Settings:
    base = dict(openai_api_key="test", model="gpt-4o")
    base.update(overrides)
    return Settings(**base)


def test_pipeline_single_chunk(pdf_factory):
    pdf = pdf_factory("doc.pdf", pages=2)
    rec = FakeRecognizer()
    pipeline = Pipeline(_settings(), recognizer=rec)

    md_path = pipeline.process(pdf)

    assert md_path.exists()
    assert rec.calls == 1
    assert read_recorded_hash(md_path) == md5_of_file(pdf)
    text = md_path.read_text(encoding="utf-8")
    assert "# Transcription 1" in text
    assert CHUNK_SEPARATOR not in text  # single chunk -> no separator


def test_pipeline_splits_and_merges(pdf_factory):
    pdf = pdf_factory("big.pdf", pages=9)
    rec = FakeRecognizer()
    pipeline = Pipeline(_settings(max_pages=3), recognizer=rec)

    md_path = pipeline.process(pdf)

    assert rec.calls > 1  # split into multiple chunks
    text = md_path.read_text(encoding="utf-8")
    assert CHUNK_SEPARATOR in text
    assert read_recorded_hash(md_path) == md5_of_file(pdf)
