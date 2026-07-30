"""OpenAI-backed OCR, Markdown formatting, and visual verification stages."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable, Protocol

import structlog
from pydantic import BaseModel, Field

from ..config import Settings
from ..monitoring import classify_llm_error, record_llm_error, record_llm_success
from .prompts import (
    MARKDOWN_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    compose_markdown_user,
    compose_ocr_user,
    compose_verify_user,
)

log = structlog.get_logger(__name__)

DeltaCallback = Callable[[str], None]

_VERIFICATION_FORMAT = {
    "type": "json_schema",
    "name": "line_corrections",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_number": {"type": "integer", "minimum": 1},
                        "replacement": {"type": "string"},
                    },
                    "required": ["line_number", "replacement"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["corrections"],
        "additionalProperties": False,
    },
}


class LineCorrection(BaseModel):
    line_number: int = Field(ge=1)
    replacement: str


class VerificationResult(BaseModel):
    corrections: list[LineCorrection]


def _redact(proxy_url: str) -> str:
    """Hide any ``user:pass@`` credentials in a proxy URL before logging it."""
    if "@" in proxy_url and "://" in proxy_url:
        scheme, rest = proxy_url.split("://", 1)
        host = rest.split("@", 1)[1]
        return f"{scheme}://***@{host}"
    return proxy_url


class DocumentProcessor(Protocol):
    """Independent stages used by the document pipeline."""

    def ocr(
        self,
        pdf_path: Path,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        ...

    def format_markdown(
        self,
        transcription: str,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        ...

    def verify(
        self,
        pdf_path: Path,
        markdown: str,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> VerificationResult:
        ...


# Compatibility name for callers that imported the old protocol.
Recognizer = DocumentProcessor


class OpenAIRecognizer:
    """Three-stage document processor backed by the OpenAI Responses API."""

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._settings.openai_api_key or None,
                http_client=self._build_http_client(),
            )
        return self._client

    def _build_http_client(self):
        proxy = self._settings.proxy.strip()
        if not proxy:
            return None
        import httpx

        if proxy.startswith("socks"):
            try:
                import socksio  # noqa: F401
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "SOCKS proxy configured but 'socksio' is not installed. "
                    "Install it with: pip install 'httpx[socks]'"
                ) from exc
        log.info("using_proxy", proxy=_redact(proxy))
        return httpx.Client(
            proxy=proxy,
            timeout=httpx.Timeout(self._settings.request_timeout, connect=15.0),
        )

    def ocr(
        self,
        pdf_path: Path,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        return self._with_retries(
            lambda: self._source_request(
                stage="ocr",
                model=self._settings.model,
                pdf_path=pdf_path,
                instructions=OCR_SYSTEM_PROMPT,
                user_text=compose_ocr_user(extra_instructions),
                on_delta=on_delta,
            )
        )

    # Compatibility method: old callers get the literal OCR stage only.
    def recognize(
        self,
        pdf_path: Path,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        return self.ocr(pdf_path, on_delta, extra_instructions)

    def format_markdown(
        self,
        transcription: str,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        model = self._settings.format_model.strip() or self._settings.model
        return self._with_retries(
            lambda: self._text_request(
                stage="format",
                model=model,
                instructions=MARKDOWN_SYSTEM_PROMPT,
                user_text=compose_markdown_user(transcription, extra_instructions),
                on_delta=on_delta,
            )
        )

    def verify(
        self,
        pdf_path: Path,
        markdown: str,
        on_delta: DeltaCallback | None = None,
        extra_instructions: str | None = None,
    ) -> VerificationResult:
        model = self._settings.verify_model.strip() or self._settings.model
        raw = self._with_retries(
            lambda: self._source_request(
                stage="verify",
                model=model,
                pdf_path=pdf_path,
                instructions=VERIFY_SYSTEM_PROMPT,
                user_text=compose_verify_user(markdown, extra_instructions),
                on_delta=on_delta,
                text_format=_VERIFICATION_FORMAT,
            )
        )
        return VerificationResult.model_validate_json(raw)

    def _source_request(
        self,
        *,
        stage: str,
        model: str,
        pdf_path: Path,
        instructions: str,
        user_text: str,
        on_delta: DeltaCallback | None,
        text_format: dict[str, object] | None = None,
    ) -> str:
        uploaded_ids: list[str] = []
        try:
            if self._settings.input_mode == "png":
                with tempfile.TemporaryDirectory(prefix="ocr-pages-") as tmp:
                    images = self._render_pages(pdf_path, Path(tmp))
                    content = [{"type": "input_text", "text": user_text}]
                    for page_number, image_path in enumerate(images, start=1):
                        file_id = self._upload(image_path, purpose="vision")
                        uploaded_ids.append(file_id)
                        content.extend(
                            [
                                {
                                    "type": "input_text",
                                    "text": f"[SOURCE PAGE {page_number}]",
                                },
                                {
                                    "type": "input_image",
                                    "file_id": file_id,
                                    "detail": self._settings.image_detail,
                                },
                            ]
                        )
                    return self._run_request(
                        stage,
                        model,
                        instructions,
                        content,
                        on_delta,
                        text_format=text_format,
                    )

            file_id = self._upload(pdf_path, purpose="user_data")
            uploaded_ids.append(file_id)
            content = [
                {"type": "input_text", "text": user_text},
                {
                    "type": "input_file",
                    "file_id": file_id,
                    "detail": self._settings.pdf_detail,
                },
            ]
            return self._run_request(
                stage,
                model,
                instructions,
                content,
                on_delta,
                text_format=text_format,
            )
        finally:
            for file_id in uploaded_ids:
                try:
                    self.client.files.delete(file_id)
                except Exception:
                    log.warning("file_delete_failed", file_id=file_id)

    def _text_request(
        self,
        *,
        stage: str,
        model: str,
        instructions: str,
        user_text: str,
        on_delta: DeltaCallback | None,
    ) -> str:
        content = [{"type": "input_text", "text": user_text}]
        return self._run_request(stage, model, instructions, content, on_delta)

    def _run_request(
        self,
        stage: str,
        model: str,
        instructions: str,
        content: list[dict[str, str]],
        on_delta: DeltaCallback | None,
        *,
        text_format: dict[str, object] | None = None,
    ) -> str:
        kwargs: dict[str, object] = {
            "model": model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "stream": True,
        }
        if self._settings.reasoning_effort:
            kwargs["reasoning"] = {"effort": self._settings.reasoning_effort}
        if text_format is not None:
            kwargs["text"] = {"format": text_format}

        log.info("llm_stage_started", stage=stage, model=model)
        try:
            stream = self.client.responses.create(**kwargs)
            text, input_tokens, output_tokens = self._collect_stream(stream, on_delta)
        except Exception as exc:
            record_llm_error("openai", model, classify_llm_error(exc))
            raise
        record_llm_success("openai", model, input_tokens, output_tokens)
        log.info("llm_stage_completed", stage=stage, model=model)
        return text

    def _upload(self, path: Path, *, purpose: str) -> str:
        with open(path, "rb") as fh:
            uploaded = self.client.files.create(file=fh, purpose=purpose)
        log.debug("uploaded_source", path=str(path), file_id=uploaded.id)
        return uploaded.id

    def _render_pages(self, pdf_path: Path, out_dir: Path) -> list[Path]:
        import pymupdf

        scale = self._settings.image_dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)
        images: list[Path] = []
        with pymupdf.open(pdf_path) as document:
            width = max(1, len(str(len(document))))
            for index, page in enumerate(document):
                dest = out_dir / f"page-{index + 1:0{width}d}.png"
                page.get_pixmap(matrix=matrix, alpha=False).save(dest)
                images.append(dest)
        log.info(
            "pdf_rendered",
            path=str(pdf_path),
            pages=len(images),
            dpi=self._settings.image_dpi,
        )
        return images

    @staticmethod
    def _collect_stream(stream, on_delta: DeltaCallback | None) -> tuple[str, int, int]:
        parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    parts.append(delta)
                    if on_delta is not None:
                        on_delta(delta)
            elif etype == "response.completed":
                usage = getattr(getattr(event, "response", None), "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
            elif etype == "response.error":
                raise RuntimeError(f"OpenAI stream error: {getattr(event, 'error', '')}")
        return "".join(parts).strip(), input_tokens, output_tokens

    def _with_retries(self, fn: Callable[[], str]) -> str:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        transient = (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        )
        attempts = self._settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except transient as exc:
                if attempt >= attempts:
                    raise
                backoff = min(2 ** (attempt - 1), 30)
                log.warning(
                    "openai_transient_error",
                    attempt=attempt,
                    backoff=backoff,
                    error=str(exc),
                )
                time.sleep(backoff)
        raise RuntimeError("unreachable")  # pragma: no cover
