"""Prometheus metrics and the standalone /metrics endpoint.

Reason labels are normalized to a small fixed vocabulary so label cardinality stays
bounded; full error text belongs in logs, never in labels.
"""

from __future__ import annotations

import time

import structlog
from prometheus_client import Counter, Gauge, start_http_server

log = structlog.get_logger(__name__)

INOTIFY_EVENTS_TOTAL = Counter(
    "app_inotify_events_total",
    "Total number of inotify events handled by the service.",
    ["event_type"],
)

SYNC_TOTAL = Counter(
    "app_sync_total",
    "Total number of sync attempts triggered by inotify events.",
    ["status", "reason"],
)

LLM_REQUESTS_TOTAL = Counter(
    "app_llm_requests_total",
    "Total number of LLM requests.",
    ["provider", "model", "status", "reason"],
)

LLM_TOKENS_TOTAL = Counter(
    "app_llm_tokens_total",
    "Total number of LLM tokens.",
    ["provider", "model", "type"],
)

LAST_SUCCESSFUL_SYNC_TIMESTAMP = Gauge(
    "app_last_successful_sync_timestamp_seconds",
    "Unix timestamp of the last successful sync.",
)


def start_monitoring(port: int) -> None:
    """Expose /metrics for Prometheus scraping on a dedicated monitoring port."""
    start_http_server(port)
    log.info("monitoring_started", port=port)


def normalize_reason(reason: str) -> str:
    allowed = {
        "none",
        "unknown",
        "timeout",
        "auth_error",
        "rate_limit",
        "db_error",
        "invalid_response",
        "provider_5xx",
    }

    if reason in allowed:
        return reason

    return "unknown"


def record_sync_success() -> None:
    SYNC_TOTAL.labels(status="success", reason="none").inc()
    LAST_SUCCESSFUL_SYNC_TIMESTAMP.set(time.time())


def record_sync_error(reason: str) -> None:
    SYNC_TOTAL.labels(
        status="error",
        reason=normalize_reason(reason),
    ).inc()


def classify_sync_error(exc: Exception) -> str:
    message = str(exc).lower()

    if "rate limit" in message:
        return "rate_limit"

    if "timeout" in message:
        return "timeout"

    if "unauthorized" in message or "forbidden" in message:
        return "auth_error"

    if "database" in message or "postgres" in message or "clickhouse" in message:
        return "db_error"

    if "invalid response" in message or "bad response" in message:
        return "invalid_response"

    return "unknown"


def record_llm_success(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    LLM_REQUESTS_TOTAL.labels(
        provider=provider,
        model=model,
        status="success",
        reason="none",
    ).inc()

    if input_tokens:
        LLM_TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            type="input",
        ).inc(input_tokens)

    if output_tokens:
        LLM_TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            type="output",
        ).inc(output_tokens)


def record_llm_error(provider: str, model: str, reason: str) -> None:
    LLM_REQUESTS_TOTAL.labels(
        provider=provider,
        model=model,
        status="error",
        reason=normalize_reason(reason),
    ).inc()


def classify_llm_error(exc: Exception) -> str:
    message = str(exc).lower()

    if "rate limit" in message or "429" in message:
        return "rate_limit"

    if "timeout" in message:
        return "timeout"

    if (
        "unauthorized" in message
        or "forbidden" in message
        or "401" in message
        or "403" in message
    ):
        return "auth_error"

    if "500" in message or "502" in message or "503" in message or "504" in message:
        return "provider_5xx"

    if "invalid response" in message or "json" in message:
        return "invalid_response"

    return "unknown"
