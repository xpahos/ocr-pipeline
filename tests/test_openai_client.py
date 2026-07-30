from __future__ import annotations

from ocr_pipeline.config import Settings
from ocr_pipeline.pipeline.openai_client import OpenAIRecognizer
from ocr_pipeline.pipeline.prompts import (
    MARKDOWN_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)


class _Uploaded:
    def __init__(self, file_id: str) -> None:
        self.id = file_id


class _Files:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.deleted: list[str] = []

    def create(self, file, purpose):
        file_id = f"file_{len(self.created) + 1}"
        self.created.append({"name": file.name, "purpose": purpose})
        return _Uploaded(file_id)

    def delete(self, file_id):
        self.deleted.append(file_id)


class _Event:
    def __init__(self, delta: str) -> None:
        self.type = "response.output_text.delta"
        self.delta = delta


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "text" in kwargs:
            return iter(
                [
                    _Event(
                        '{"corrections":[{"line_number":2,'
                        '"replacement":"corrected"}]}'
                    )
                ]
            )
        return iter([_Event("[PAGE 1]"), _Event("\nbody")])


class StubClient:
    def __init__(self) -> None:
        self.files = _Files()
        self.responses = _Responses()


def _recognizer(stub: StubClient, **overrides) -> OpenAIRecognizer:
    settings = Settings(openai_api_key="x", **overrides)
    return OpenAIRecognizer(settings, client=stub)


def test_ocr_sends_pdf_at_high_detail_and_cleans_up(pdf_factory):
    pdf = pdf_factory("a.pdf")
    stub = StubClient()
    out = _recognizer(stub).ocr(pdf)

    assert out == "[PAGE 1]\nbody"
    kwargs = stub.responses.calls[0]
    assert kwargs["instructions"] == OCR_SYSTEM_PROMPT
    content = kwargs["input"][0]["content"]
    assert "<task>" in content[0]["text"]
    assert content[1] == {
        "type": "input_file",
        "file_id": "file_1",
        "detail": "high",
    }
    assert stub.files.created[0]["purpose"] == "user_data"
    assert stub.files.deleted == ["file_1"]


def test_format_is_text_only_and_verify_sees_source(pdf_factory):
    pdf = pdf_factory("a.pdf")
    stub = StubClient()
    recognizer = _recognizer(stub)

    recognizer.format_markdown("[PAGE 1]\nbody")
    verification = recognizer.verify(pdf, "[PAGE 1]\n# body")

    format_call, verify_call = stub.responses.calls
    assert format_call["instructions"] == MARKDOWN_SYSTEM_PROMPT
    assert len(format_call["input"][0]["content"]) == 1
    assert "<transcription>" in format_call["input"][0]["content"][0]["text"]
    assert verify_call["instructions"] == VERIFY_SYSTEM_PROMPT
    assert verify_call["input"][0]["content"][1]["type"] == "input_file"
    assert verify_call["text"]["format"]["type"] == "json_schema"
    assert verification.corrections[0].replacement == "corrected"


def test_corrections_are_present_in_each_stage(pdf_factory):
    pdf = pdf_factory("a.pdf")
    stub = StubClient()
    recognizer = _recognizer(stub)

    recognizer.ocr(pdf, extra_instructions="The name is Пётр")
    recognizer.format_markdown("text", extra_instructions="The name is Пётр")
    recognizer.verify(pdf, "text", extra_instructions="The name is Пётр")

    for call in stub.responses.calls:
        text_part = call["input"][0]["content"][0]["text"]
        assert "<corrections>" in text_part
        assert "The name is Пётр" in text_part


def test_png_mode_renders_and_uploads_page_images(pdf_factory):
    pdf = pdf_factory("a.pdf", pages=2)
    stub = StubClient()

    _recognizer(stub, input_mode="png", image_dpi=72).ocr(pdf)

    assert [item["purpose"] for item in stub.files.created] == ["vision", "vision"]
    content = stub.responses.calls[0]["input"][0]["content"]
    image_parts = [part for part in content if part["type"] == "input_image"]
    assert len(image_parts) == 2
    assert all(part["detail"] == "original" for part in image_parts)
    assert stub.files.deleted == ["file_1", "file_2"]


def test_reasoning_effort_and_stage_specific_models(pdf_factory):
    pdf = pdf_factory("a.pdf")
    stub = StubClient()
    recognizer = _recognizer(
        stub,
        model="gpt-5.6-sol",
        format_model="gpt-5.6-terra",
        verify_model="gpt-5.6-sol",
        reasoning_effort="high",
    )

    recognizer.ocr(pdf)
    recognizer.format_markdown("text")
    recognizer.verify(pdf, "text")

    assert [call["model"] for call in stub.responses.calls] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    ]
    assert all(
        call["reasoning"] == {"effort": "high"} for call in stub.responses.calls
    )
