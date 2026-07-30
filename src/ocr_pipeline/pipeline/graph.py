"""Assemble and run the LangGraph transcription pipeline."""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from . import nodes
from .nodes import PipelineState
from .openai_client import DocumentProcessor, OpenAIRecognizer


def build_graph(settings: Settings, processor: DocumentProcessor):
    """Build the validate -> split -> OCR -> format -> verify -> merge -> write graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("validate", functools.partial(nodes.validate_node, settings=settings))
    graph.add_node("split", functools.partial(nodes.split_node, settings=settings))
    graph.add_node(
        "ocr",
        functools.partial(nodes.ocr_node, settings=settings, processor=processor),
    )
    graph.add_node(
        "format",
        functools.partial(nodes.format_node, settings=settings, processor=processor),
    )
    graph.add_node(
        "verify",
        functools.partial(nodes.verify_node, settings=settings, processor=processor),
    )
    graph.add_node("merge", functools.partial(nodes.merge_node, settings=settings))
    graph.add_node("write", functools.partial(nodes.write_node, settings=settings))

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "split")
    graph.add_edge("split", "ocr")
    graph.add_edge("ocr", "format")
    graph.add_edge("format", "verify")
    graph.add_edge("verify", "merge")
    graph.add_edge("merge", "write")
    graph.add_edge("write", END)

    return graph.compile()


class Pipeline:
    """Stateful, reusable pipeline: compiles the graph once, runs it per PDF."""

    def __init__(
        self,
        settings: Settings,
        processor: DocumentProcessor | None = None,
        *,
        recognizer: DocumentProcessor | None = None,
    ) -> None:
        self.settings = settings
        # ``recognizer`` remains accepted for callers using the old keyword.
        self.processor = processor or recognizer or OpenAIRecognizer(settings)
        self.recognizer = self.processor
        self._compiled = build_graph(settings, self.processor)

    def process(self, pdf_path: Path) -> Path:
        """Transcribe one PDF and return the path of the written ``.md``.

        Splitting may create temporary chunk files; they live in a TemporaryDirectory
        that is cleaned up once the run completes (success or failure).
        """
        with tempfile.TemporaryDirectory(prefix="ocr-chunks-") as work_dir:
            result = self._compiled.invoke(
                {"pdf_path": str(pdf_path), "work_dir": work_dir}
            )
        return Path(result["md_path"])
