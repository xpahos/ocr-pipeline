"""Upload a PDF (chunk) and stream its transcription from the OpenAI Responses API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Protocol

import structlog

from ..config import Settings
from .prompts import SYSTEM_PROMPT, USER_PROMPT

log = structlog.get_logger(__name__)

# Callback invoked with each streamed text delta (for progress / incremental writes).
DeltaCallback = Callable[[str], None]


def _redact(proxy_url: str) -> str:
    """Hide any ``user:pass@`` credentials in a proxy URL before logging it."""
    if "@" in proxy_url and "://" in proxy_url:
        scheme, rest = proxy_url.split("://", 1)
        host = rest.split("@", 1)[1]
        return f"{scheme}://***@{host}"
    return proxy_url


class Recognizer(Protocol):
    """Transcribes a single PDF (already within OpenAI's per-request limits)."""

    def recognize(self, pdf_path: Path, on_delta: DeltaCallback | None = None) -> str:
        ...


class OpenAIRecognizer:
    """Recognizer backed by the OpenAI Responses API with streaming output."""

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def client(self):  # lazily construct so importing doesn't require an API key
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._settings.openai_api_key or None,
                http_client=self._build_http_client(),
            )
        return self._client

    def _build_http_client(self):
        """Build an httpx client routed through the configured proxy, or None.

        Supports SOCKS5 (``socks5://`` / ``socks5h://``) and HTTP(S) proxies. SOCKS
        requires the ``socksio`` package (installed via the ``httpx[socks]`` extra).
        """
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

    def recognize(self, pdf_path: Path, on_delta: DeltaCallback | None = None) -> str:
        return self._with_retries(lambda: self._recognize_once(pdf_path, on_delta))

    # -- internals ---------------------------------------------------------------

    def _recognize_once(self, pdf_path: Path, on_delta: DeltaCallback | None) -> str:
        client = self.client
        with open(pdf_path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="user_data")
        file_id = uploaded.id
        log.debug("uploaded_pdf", path=str(pdf_path), file_id=file_id)
        try:
            stream = client.responses.create(
                model=self._settings.model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_PROMPT},
                            {"type": "input_file", "file_id": file_id},
                        ],
                    }
                ],
                stream=True,
            )
            return self._collect_stream(stream, on_delta)
        finally:
            try:
                client.files.delete(file_id)
            except Exception:  # cleanup is best-effort
                log.warning("file_delete_failed", file_id=file_id)

    @staticmethod
    def _collect_stream(stream, on_delta: DeltaCallback | None) -> str:
        parts: list[str] = []
        for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    parts.append(delta)
                    if on_delta is not None:
                        on_delta(delta)
            elif etype == "response.error":
                raise RuntimeError(f"OpenAI stream error: {getattr(event, 'error', '')}")
        return "".join(parts).strip()

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
