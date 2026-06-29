from __future__ import annotations

from ocr_pipeline.config import Settings
from ocr_pipeline.pipeline.openai_client import OpenAIRecognizer
from ocr_pipeline.pipeline.prompts import SYSTEM_PROMPT


class _Uploaded:
    id = "file_123"


class _Files:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def create(self, file, purpose):
        return _Uploaded()

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
        return iter([_Event("# Title"), _Event("\n\nbody")])


class StubClient:
    def __init__(self) -> None:
        self.files = _Files()
        self.responses = _Responses()


def _recognizer(stub: StubClient) -> OpenAIRecognizer:
    return OpenAIRecognizer(Settings(openai_api_key="x"), client=stub)


def test_recognize_streams_and_sends_static_system(pdf_factory):
    pdf = pdf_factory("a.pdf")
    stub = StubClient()
    out = _recognizer(stub).recognize(pdf)

    assert out == "# Title\n\nbody"  # deltas concatenated
    kwargs = stub.responses.calls[0]
    assert kwargs["instructions"] == SYSTEM_PROMPT  # system prompt is static

    content = kwargs["input"][0]["content"]
    text_part = content[0]["text"]
    assert "<task>" in text_part
    assert "<corrections>" not in text_part
    assert content[1] == {"type": "input_file", "file_id": "file_123"}
    assert stub.files.deleted == ["file_123"]  # uploaded file cleaned up


def test_corrections_go_to_user_message_not_system(pdf_factory):
    pdf = pdf_factory("a.pdf")
    stub = StubClient()
    _recognizer(stub).recognize(pdf, extra_instructions="Fix the date on page 2")

    kwargs = stub.responses.calls[0]
    # System prompt is unchanged regardless of corrections...
    assert kwargs["instructions"] == SYSTEM_PROMPT
    # ...and the corrections live in the user message.
    text_part = kwargs["input"][0]["content"][0]["text"]
    assert "<corrections>" in text_part
    assert "Fix the date on page 2" in text_part
