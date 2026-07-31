from __future__ import annotations

from ocr_pipeline.pipeline.prompts import (
    MARKDOWN_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    compose_markdown_user,
    compose_ocr_user,
    compose_verify_user,
)


def test_ocr_prompt_is_literal_not_markdown():
    assert "literal OCR" in OCR_SYSTEM_PROMPT
    assert "Do not infer Markdown structure" in OCR_SYSTEM_PROMPT
    assert "[PAGE N]" in OCR_SYSTEM_PROMPT
    assert "[illegible]" in OCR_SYSTEM_PROMPT
    assert "[BJ:M]" in OCR_SYSTEM_PROMPT
    assert "[BJ-CONT]" in OCR_SYSTEM_PROMPT
    assert "never replace them with arrows" in OCR_SYSTEM_PROMPT


def test_markdown_prompt_formats_without_retranscribing():
    assert "Obsidian-flavored Markdown" in MARKDOWN_SYSTEM_PROMPT
    assert "[BJ:M]" in MARKDOWN_SYSTEM_PROMPT
    assert "[BJ-CONT]" in MARKDOWN_SYSTEM_PROMPT
    assert "protected data" in MARKDOWN_SYSTEM_PROMPT
    assert "Never re-read, correct, invent" in MARKDOWN_SYSTEM_PROMPT


def test_verify_prompt_checks_high_risk_fields():
    for value in ("numbers", "dates", "names", "table rows"):
        assert value in VERIFY_SYSTEM_PROMPT
    assert "complete corrected content of that single line" in VERIFY_SYSTEM_PROMPT


def test_user_messages_keep_payloads_and_corrections_separate():
    ocr = compose_ocr_user("The name is Пётр")
    formatted = compose_markdown_user("[PAGE 1]\ntext")
    verified = compose_verify_user("[PAGE 1]\n# text")

    assert "<corrections>" in ocr and "Пётр" in ocr
    assert "<transcription>" in formatted
    assert "<candidate_lines>" in verified
    assert "<corrections>" not in compose_ocr_user("   ")
