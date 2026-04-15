from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from chatbox.domain.errors import ParsingError
from chatbox.ingestion.parsers.types import ParsedText


class DocxParser:
    def __init__(self, document_factory: Callable[[str], Any] | None = None) -> None:
        self._document_factory = document_factory

    def parse(self, file_path: Path) -> ParsedText:
        document_factory = self._document_factory
        if document_factory is None:
            try:
                from docx import Document as DocxDocument  # type: ignore
            except ImportError as exc:
                raise ParsingError("python-docx is required for DOCX parsing") from exc
            document_factory = DocxDocument

        try:
            document = document_factory(str(file_path))
            lines = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            text = "\n".join(lines)
            return ParsedText(
                file_type="docx",
                text=text,
                metadata={"paragraph_count": len(lines)},
            )
        except Exception as exc:  # pragma: no cover - defensive parser boundary
            raise ParsingError(f"failed to parse DOCX: {file_path}") from exc


__all__ = ["DocxParser", "ParsedText"]
