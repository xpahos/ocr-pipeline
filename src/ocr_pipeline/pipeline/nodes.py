"""LangGraph node functions for the transcription pipeline.

Nodes raise on failure; the serial worker that invokes the graph catches exceptions and
leaves the PDF un-transcribed so it is retried later.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import structlog

from ..config import Settings
from ..hashing import md5_of_file
from ..mdfile import md_path_for, read_instructions, write_md
from ..pdfutil import split_pdf
from .openai_client import Recognizer

log = structlog.get_logger(__name__)

# Separator between merged chunk transcriptions.
CHUNK_SEPARATOR = "\n\n---\n\n"


class PipelineState(TypedDict, total=False):
    pdf_path: str
    work_dir: str
    pdf_hash: str
    instructions: str | None
    chunks: list[str]
    parts: list[str]
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


def recognize_node(
    state: PipelineState, *, settings: Settings, recognizer: Recognizer
) -> PipelineState:
    chunks = state["chunks"]
    instructions = state.get("instructions")
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        log.info("recognize_chunk", chunk=idx, total=len(chunks))
        text = recognizer.recognize(Path(chunk), extra_instructions=instructions)
        parts.append(text)
    return {"parts": parts}


def merge_node(state: PipelineState, *, settings: Settings) -> PipelineState:
    parts = [p.strip() for p in state["parts"] if p.strip()]
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
