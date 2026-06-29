from __future__ import annotations

from ocr_pipeline.pipeline.prompts import SYSTEM_PROMPT, compose_user


def test_system_prompt_is_structured():
    for tag in ("<role>", "</role>", "<style>", "</style>", "<rules>", "</rules>"):
        assert tag in SYSTEM_PROMPT


def test_system_prompt_has_core_rules():
    # Obsidian-flavored target, no translation, and the first-column Bullet-Journal rule.
    assert "Obsidian-flavored" in SYSTEM_PROMPT
    assert "Do NOT translate" in SYSTEM_PROMPT
    assert "FIRST COLUMN" in SYSTEM_PROMPT
    assert "Bullet-Journal" in SYSTEM_PROMPT
    for mark in ("`x`", "`>`", "`!`"):
        assert mark in SYSTEM_PROMPT


def test_compose_user_without_corrections():
    text = compose_user(None)
    assert "<task>" in text and "</task>" in text
    assert "<output>" in text and "</output>" in text
    assert "<corrections>" not in text


def test_compose_user_blank_corrections_omits_block():
    assert "<corrections>" not in compose_user("   ")


def test_compose_user_with_corrections():
    text = compose_user("Page 2 is a UML diagram.")
    assert "<corrections>" in text and "</corrections>" in text
    assert "Page 2 is a UML diagram." in text
    # Corrections are framed as overriding the base rules.
    assert "higher priority" in text
