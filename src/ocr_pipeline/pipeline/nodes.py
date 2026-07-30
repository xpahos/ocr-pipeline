"""LangGraph node functions for the transcription pipeline.

Nodes raise on failure; the serial worker that invokes the graph catches exceptions and
leaves the PDF un-transcribed so it is retried later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import structlog

from ..config import Settings
from ..hashing import md5_of_file
from ..mdfile import md_path_for, read_instructions, write_md
from ..pdfutil import inspect, split_pdf
from .openai_client import DocumentProcessor, VerificationResult

log = structlog.get_logger(__name__)

# Separator between merged chunk transcriptions.
CHUNK_SEPARATOR = "\n\n---\n\n"
_PAGE_MARKER_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)
_UNCERTAINTY_RE = re.compile(r"\[(?:\?|illegible)\]", re.IGNORECASE)


class PipelineState(TypedDict, total=False):
    pdf_path: str
    work_dir: str
    pdf_hash: str
    instructions: str | None
    chunks: list[str]
    ocr_parts: list[str]
    markdown_parts: list[str]
    verified_parts: list[str]
    body: str
    md_path: str


def validate_node(state: PipelineState, *, settings: Settings) -> PipelineState:
    pdf_path = Path(state["pdf_path"])
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF disappeared before processing: {pdf_path}")
    # Carry forward any human correction instructions from the existing .md so they feed
    # this pass and are re-appended afterwards.
    instructions = read_instructions(md_path_for(pdf_path))
    if instructions:
        log.info("using_instructions", pdf=str(pdf_path), chars=len(instructions))
    # Hash the current bytes up front; the write step reuses it so the recorded MD5
    # matches the bytes we actually transcribed.
    return {"pdf_hash": md5_of_file(pdf_path), "instructions": instructions}


def split_node(state: PipelineState, *, settings: Settings) -> PipelineState:
    pdf_path = Path(state["pdf_path"])
    work_dir = Path(state["work_dir"])
    chunks = split_pdf(
        pdf_path,
        work_dir,
        max_pages=settings.max_pages,
        max_bytes=settings.max_bytes,
    )
    if len(chunks) > 1:
        log.info("pdf_split", path=str(pdf_path), chunks=len(chunks))
    return {"chunks": [str(c) for c in chunks]}


def ocr_node(
    state: PipelineState, *, settings: Settings, processor: DocumentProcessor
) -> PipelineState:
    chunks = state["chunks"]
    instructions = state.get("instructions")
    parts: list[str] = []
    page_offset = 0
    for idx, chunk in enumerate(chunks, start=1):
        log.info("ocr_chunk", chunk=idx, total=len(chunks))
        chunk_path = Path(chunk)
        text = processor.ocr(chunk_path, extra_instructions=instructions)
        parts.append(
            _PAGE_MARKER_RE.sub(
                lambda match: f"[PAGE {int(match.group(1)) + page_offset}]",
                text,
            )
        )
        page_offset += inspect(chunk_path).page_count
    return {"ocr_parts": parts}


def format_node(
    state: PipelineState, *, settings: Settings, processor: DocumentProcessor
) -> PipelineState:
    if not settings.formatting_enabled:
        return {"markdown_parts": state["ocr_parts"]}

    instructions = state.get("instructions")
    parts: list[str] = []
    for idx, transcription in enumerate(state["ocr_parts"], start=1):
        log.info("format_chunk", chunk=idx, total=len(state["ocr_parts"]))
        parts.append(
            processor.format_markdown(
                transcription, extra_instructions=instructions
            )
        )
    return {"markdown_parts": parts}


def verify_node(
    state: PipelineState, *, settings: Settings, processor: DocumentProcessor
) -> PipelineState:
    if settings.verify_mode == "off":
        return {"verified_parts": state["markdown_parts"]}

    instructions = state.get("instructions")
    chunks = state["chunks"]
    parts: list[str] = []
    for idx, (chunk, markdown) in enumerate(
        zip(chunks, state["markdown_parts"], strict=True), start=1
    ):
        should_verify = (
            settings.verify_mode == "always"
            or bool(instructions and instructions.strip())
            or bool(_UNCERTAINTY_RE.search(markdown))
        )
        if not should_verify:
            log.info("verify_chunk_skipped", chunk=idx, reason="no_uncertainty")
            parts.append(markdown)
            continue
        log.info("verify_chunk", chunk=idx, total=len(chunks))
        result = processor.verify(
            Path(chunk), markdown, extra_instructions=instructions
        )
        parts.append(apply_line_corrections(markdown, result))
    return {"verified_parts": parts}


def apply_line_corrections(markdown: str, result: VerificationResult) -> str:
    """Apply verifier replacements without asking the model to reproduce the document."""
    lines = markdown.splitlines()
    seen: set[int] = set()
    for correction in result.corrections:
        line_number = correction.line_number
        if line_number in seen:
            raise ValueError(f"Verifier returned duplicate line {line_number}")
        if line_number > len(lines):
            raise ValueError(
                f"Verifier returned line {line_number}, but candidate has {len(lines)} lines"
            )
        if "\n" in correction.replacement or "\r" in correction.replacement:
            raise ValueError(
                f"Verifier replacement for line {line_number} contains a newline"
            )
        seen.add(line_number)
        lines[line_number - 1] = correction.replacement
    return "\n".join(lines)


def merge_node(state: PipelineState, *, settings: Settings) -> PipelineState:
    parts = [p.strip() for p in state["verified_parts"] if p.strip()]
    return {"body": CHUNK_SEPARATOR.join(parts)}


def write_node(state: PipelineState, *, settings: Settings) -> PipelineState:
    pdf_path = Path(state["pdf_path"])
    md_path = write_md(
        pdf_path,
        state["body"],
        pdf_hash=state["pdf_hash"],
        instructions=state.get("instructions"),
    )
    log.info("wrote_md", pdf=str(pdf_path), md=str(md_path))
    return {"md_path": str(md_path)}
