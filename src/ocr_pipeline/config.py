"""Service configuration, loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the OCR pipeline service."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI — read without the OCR_ prefix, matching the conventional name.
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    vault_root: Path = Field(default=Path("."))
    model: str = Field(default="gpt-4o")

    # Optional proxy for all OpenAI traffic, e.g. a corporate SOCKS5 proxy:
    #   socks5h://user:pass@proxy.corp:1080  (socks5h => DNS resolved by the proxy)
    # Also accepts http(s):// proxies. Empty = direct connection.
    proxy: str = Field(default="")
    # Read timeout (seconds) for OpenAI calls; transcription streams can run long.
    request_timeout: float = Field(default=600.0, gt=0)

    # Per-request OpenAI limits; PDFs above either bound are auto-split.
    max_pages: int = Field(default=100, gt=0)
    max_bytes: int = Field(default=33_000_000, gt=0)

    debounce_seconds: float = Field(default=1.5, ge=0)
    # NoDecode: keep pydantic-settings from JSON-parsing the env value so the validator
    # below can split a plain comma-separated string (e.g. ".obsidian/**,.trash/**").
    ignore_globs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [".obsidian/**", ".trash/**"]
    )
    log_level: str = Field(default="INFO")
    max_retries: int = Field(default=3, ge=0)

    @field_validator("ignore_globs", mode="before")
    @classmethod
    def _split_globs(cls, value: object) -> object:
        """Allow OCR_IGNORE_GLOBS to be a comma-separated string."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("vault_root")
    @classmethod
    def _expand_root(cls, value: Path) -> Path:
        return value.expanduser()


def load_settings() -> Settings:
    """Load settings from the environment / .env file."""
    return Settings()
