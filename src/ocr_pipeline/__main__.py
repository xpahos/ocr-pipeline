"""CLI entry point: load settings, configure logging, run the service."""

from __future__ import annotations

import signal
import sys

import structlog

from .config import load_settings
from .logging_setup import configure_logging
from .service import Service

log = structlog.get_logger(__name__)


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level)

    if not settings.openai_api_key:
        log.error("missing_api_key", hint="set OPENAI_API_KEY in the environment or .env")
        return 2

    service = Service(settings)

    def _on_signal(signum, _frame):
        log.info("signal_received", signum=signum)
        service.stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        service.run()
    except (NotADirectoryError, FileNotFoundError) as exc:
        log.error("startup_failed", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
