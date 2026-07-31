from __future__ import annotations

from pathlib import Path

from ocr_pipeline.config import Settings
from ocr_pipeline.hashing import md5_of_file
from ocr_pipeline.mdfile import read_recorded_hash
from ocr_pipeline.pipeline.graph import Pipeline
from ocr_pipeline.pipeline.nodes import (
    CHUNK_SEPARATOR,
    apply_line_corrections,
    render_bullet_journal_blocks,
)
from ocr_pipeline.pipeline.openai_client import LineCorrection, VerificationResult


class FakeProcessor:
    def __init__(self) -> None:
        self.ocr_calls = 0
        self.format_calls = 0
        self.verify_calls = 0
        self.instructions_seen: list[str | None] = []

    def ocr(self, pdf_path: Path, on_delta=None, extra_instructions=None) -> str:
        self.ocr_calls += 1
        self.instructions_seen.append(extra_instructions)
        return f"[PAGE 1]\nliteral body {self.ocr_calls}"

    def format_markdown(
        self, transcription: str, on_delta=None, extra_instructions=None
    ) -> str:
        self.format_calls += 1
        self.instructions_seen.append(extra_instructions)
        return transcription.replace("literal body", "# Formatted")

    def verify(
        self, pdf_path: Path, markdown: str, on_delta=None, extra_instructions=None
    ) -> VerificationResult:
        self.verify_calls += 1
        self.instructions_seen.append(extra_instructions)
        return VerificationResult(corrections=[])


def _settings(**overrides) -> Settings:
    base = dict(openai_api_key="test", model="gpt-5.6-sol")
    base.update(overrides)
    return Settings(**base)


def test_pipeline_runs_all_stages(pdf_factory):
    pdf = pdf_factory("doc.pdf", pages=2)
    processor = FakeProcessor()
    pipeline = Pipeline(_settings(verify_mode="always"), processor=processor)

    md_path = pipeline.process(pdf)

    assert md_path.exists()
    assert processor.ocr_calls == 1
    assert processor.format_calls == 1
    assert processor.verify_calls == 1
    assert read_recorded_hash(md_path) == md5_of_file(pdf)
    text = md_path.read_text(encoding="utf-8")
    assert "# Formatted 1" in text
    assert CHUNK_SEPARATOR not in text


def test_pipeline_passes_and_preserves_instructions(pdf_factory):
    from ocr_pipeline.mdfile import read_instructions, write_md

    pdf = pdf_factory("doc.pdf", pages=1)
    write_md(pdf, "old transcription", instructions="Page 1 'Пётр' is a name.")

    processor = FakeProcessor()
    pipeline = Pipeline(_settings(), processor=processor)
    md_path = pipeline.process(pdf)

    assert processor.ocr_calls == 0
    assert processor.format_calls == 0
    assert processor.verify_calls == 1
    assert processor.instructions_seen == ["Page 1 'Пётр' is a name."]
    assert read_instructions(md_path) == "Page 1 'Пётр' is a name."
    text = md_path.read_text(encoding="utf-8")
    assert "old transcription" in text
    assert "## OCR Instructions" in text


def test_pipeline_splits_renumbers_and_merges(pdf_factory):
    pdf = pdf_factory("big.pdf", pages=9)
    processor = FakeProcessor()
    pipeline = Pipeline(_settings(max_pages=3), processor=processor)

    md_path = pipeline.process(pdf)

    assert processor.ocr_calls > 1
    text = md_path.read_text(encoding="utf-8")
    assert CHUNK_SEPARATOR in text
    assert "[PAGE " not in text
    assert read_recorded_hash(md_path) == md5_of_file(pdf)


def test_stages_can_be_disabled(pdf_factory):
    pdf = pdf_factory("doc.pdf")
    processor = FakeProcessor()
    pipeline = Pipeline(
        _settings(formatting_enabled=False, verify_mode="off"),
        processor=processor,
    )

    text = pipeline.process(pdf).read_text(encoding="utf-8")

    assert processor.ocr_calls == 1
    assert processor.format_calls == 0
    assert processor.verify_calls == 0
    assert "literal body" in text


def test_uncertain_mode_verifies_only_marked_chunks(pdf_factory):
    class UncertainProcessor(FakeProcessor):
        def ocr(self, pdf_path: Path, on_delta=None, extra_instructions=None) -> str:
            self.ocr_calls += 1
            self.instructions_seen.append(extra_instructions)
            return "[PAGE 1]\n18[?]4"

        def format_markdown(
            self, transcription: str, on_delta=None, extra_instructions=None
        ) -> str:
            self.format_calls += 1
            self.instructions_seen.append(extra_instructions)
            return transcription

        def verify(
            self, pdf_path: Path, markdown: str, on_delta=None, extra_instructions=None
        ) -> VerificationResult:
            self.verify_calls += 1
            self.instructions_seen.append(extra_instructions)
            return VerificationResult(
                corrections=[LineCorrection(line_number=2, replacement="1884")]
            )

    pdf = pdf_factory("uncertain.pdf")
    processor = UncertainProcessor()

    text = Pipeline(_settings(), processor=processor).process(pdf).read_text(
        encoding="utf-8"
    )

    assert processor.verify_calls == 1
    assert "1884" in text
    assert "18[?]4" not in text


def test_uncertain_mode_skips_clean_chunk(pdf_factory):
    pdf = pdf_factory("clean.pdf")
    processor = FakeProcessor()

    Pipeline(_settings(), processor=processor).process(pdf)

    assert processor.ocr_calls == 1
    assert processor.format_calls == 1
    assert processor.verify_calls == 0


def test_line_corrections_reject_invalid_or_duplicate_lines():
    duplicate = VerificationResult(
        corrections=[
            LineCorrection(line_number=1, replacement="first"),
            LineCorrection(line_number=1, replacement="again"),
        ]
    )
    out_of_range = VerificationResult(
        corrections=[LineCorrection(line_number=3, replacement="missing")]
    )

    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        apply_line_corrections("original", duplicate)
    with pytest.raises(ValueError, match="candidate has 1 lines"):
        apply_line_corrections("original", out_of_range)


def test_bullet_journal_marks_are_rendered_deterministically():
    transcription = (
        "[PAGE 1]\n"
        "[BJ:>] moved\n"
        "[BJ-CONT] continuation one\n"
        "[BJ-CONT] continuation two\n"
        "[BJ:o] event\n\n"
        "normal"
    )
    formatted = transcription

    result = render_bullet_journal_blocks(transcription, formatted)

    assert result.count("| Mark | Entry |") == 1
    assert (
        "| `>` | moved<br>continuation one<br>continuation two |" in result
    )
    assert "| `o` | event |" in result
    assert "→" not in result
    assert "[BJ:" not in result
    assert "[BJ-CONT]" not in result


def test_bullet_journal_mark_changes_fail_before_write():
    import pytest

    with pytest.raises(ValueError, match="changed, dropped, or reordered"):
        render_bullet_journal_blocks("[BJ:>] moved", "[BJ:→] moved")
    with pytest.raises(ValueError, match="invalid Bullet-Journal marks"):
        render_bullet_journal_blocks("[BJ:1] item", "[BJ:1] item")
    with pytest.raises(ValueError, match="continuation has no preceding"):
        render_bullet_journal_blocks("[BJ-CONT] orphan", "[BJ-CONT] orphan")


def test_formatter_regression_cannot_overwrite_existing_markdown(pdf_factory):
    from ocr_pipeline.mdfile import write_md

    class ArrowProcessor(FakeProcessor):
        def ocr(self, pdf_path: Path, on_delta=None, extra_instructions=None) -> str:
            return "[PAGE 1]\n[BJ:>] moved"

        def format_markdown(
            self, transcription: str, on_delta=None, extra_instructions=None
        ) -> str:
            return "[PAGE 1]\n[BJ:→] moved"

    import pytest

    pdf = pdf_factory("history.pdf")
    md = write_md(pdf, "| Mark | Entry |\n| --- | --- |\n| `>` | historic |")
    before = md.read_text(encoding="utf-8")
    pdf.write_bytes(pdf.read_bytes() + b"\n% changed")

    with pytest.raises(ValueError, match="changed, dropped, or reordered"):
        Pipeline(_settings(), processor=ArrowProcessor()).process(pdf)

    assert md.read_text(encoding="utf-8") == before
