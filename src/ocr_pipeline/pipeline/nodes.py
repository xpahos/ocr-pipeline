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
from ..mdfile import (
    md_path_for,
    read_body,
    read_instructions,
    read_recorded_hash,
    write_md,
)
from ..pdfutil import inspect, split_pdf
from .openai_client import DocumentProcessor, VerificationResult

log = structlog.get_logger(__name__)

# Separator between merged chunk transcriptions.
CHUNK_SEPARATOR = "\n\n---\n\n"
_PAGE_MARKER_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)
_UNCERTAINTY_RE = re.compile(r"\[(?:\?|illegible)\]", re.IGNORECASE)
_BJ_MARK_RE = re.compile(r"^\[BJ:([^\]\r\n]+)\](?:[ \t]+(.*))?$")
_BJ_CONT_RE = re.compile(r"^\[BJ-CONT\](?:[ \t]+(.*))?$")
_BJ_ALLOWED_MARKS = frozenset({"x", ">", "<", "-", "?", "!", "*", "o"})


class PipelineState(TypedDict, total=False):
    pdf_path: str
    work_dir: str
    pdf_hash: str
    instructions: str | None
    existing_body: str | None
    reuse_existing: bool
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
    pdf_hash = md5_of_file(pdf_path)
    md_path = md_path_for(pdf_path)
    existing_body = read_body(md_path)
    reuse_existing = bool(
        instructions
        and read_recorded_hash(md_path) == pdf_hash
        and existing_body
    )
    if reuse_existing:
        log.info("preserving_existing_markdown", pdf=str(pdf_path))
    return {
        "pdf_hash": pdf_hash,
        "instructions": instructions,
        "existing_body": existing_body if reuse_existing else None,
        "reuse_existing": reuse_existing,
    }


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
    if state.get("reuse_existing"):
        return {"ocr_parts": []}
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
    if state.get("reuse_existing"):
        return {"markdown_parts": [state["existing_body"]]}
    if not settings.formatting_enabled:
        return {"markdown_parts": state["ocr_parts"]}

    instructions = state.get("instructions")
    parts: list[str] = []
    for idx, transcription in enumerate(state["ocr_parts"], start=1):
        log.info("format_chunk", chunk=idx, total=len(state["ocr_parts"]))
        formatted = processor.format_markdown(
            transcription, extra_instructions=instructions
        )
        parts.append(render_bullet_journal_blocks(transcription, formatted))
    return {"markdown_parts": parts}


def verify_node(
    state: PipelineState, *, settings: Settings, processor: DocumentProcessor
) -> PipelineState:
    if state.get("reuse_existing"):
        markdown = state["markdown_parts"][0]
        result = processor.verify(
            Path(state["pdf_path"]),
            markdown,
            extra_instructions=state.get("instructions"),
        )
        return {"verified_parts": [apply_line_corrections(markdown, result)]}
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


def render_bullet_journal_blocks(transcription: str, markdown: str) -> str:
    """Render protected OCR mark sentinels as stable Markdown tables."""
    def sentinel_sequence(text: str) -> list[str]:
        sequence: list[str] = []
        for line in text.splitlines():
            if match := _BJ_MARK_RE.fullmatch(line):
                sequence.append(f"mark:{match.group(1)}")
            elif _BJ_CONT_RE.fullmatch(line):
                sequence.append("continuation")
        return sequence

    expected = sentinel_sequence(transcription)
    found = sentinel_sequence(markdown)
    invalid = [
        token.removeprefix("mark:")
        for token in expected
        if token.startswith("mark:")
        and token.removeprefix("mark:") not in _BJ_ALLOWED_MARKS
    ]
    if invalid:
        raise ValueError(
            f"OCR returned invalid Bullet-Journal marks: {', '.join(invalid)}"
        )
    if found != expected:
        raise ValueError(
            "Markdown formatter changed, dropped, or reordered Bullet-Journal marks"
        )

    lines = markdown.splitlines()
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        match = _BJ_MARK_RE.fullmatch(lines[index])
        if not match:
            rendered.append(lines[index])
            index += 1
            continue

        rendered.extend(["| Mark | Entry |", "|---|---|"])
        while index < len(lines):
            match = _BJ_MARK_RE.fullmatch(lines[index])
            if not match:
                break
            mark = match.group(1).replace("|", r"\|")
            entry_lines = [(match.group(2) or "").replace("|", r"\|")]
            index += 1
            while index < len(lines):
                continuation = _BJ_CONT_RE.fullmatch(lines[index])
                if not continuation:
                    break
                entry_lines.append(
                    (continuation.group(1) or "").replace("|", r"\|")
                )
                index += 1
            rendered.append(f"| `{mark}` | {'<br>'.join(entry_lines)} |")

    if any(_BJ_CONT_RE.fullmatch(line) for line in rendered):
        raise ValueError("Bullet-Journal continuation has no preceding marked entry")
    return "\n".join(rendered)


def merge_node(state: PipelineState, *, settings: Settings) -> PipelineState:
    parts = [
        _PAGE_MARKER_RE.sub("", p).strip()
        for p in state["verified_parts"]
        if p.strip()
    ]
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
