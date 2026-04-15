from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from chatbox.domain.errors import ParsingError
from chatbox.ingestion.parsers.types import ParsedText


class PdfParser:
    def __init__(self, reader_factory: Callable[[str], Any] | None = None) -> None:
        self._reader_factory = reader_factory

    def parse(self, file_path: Path) -> ParsedText:
        reader_factory = self._reader_factory
        if reader_factory is None:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError as exc:
                raise ParsingError("pypdf is required for PDF parsing") from exc
            reader_factory = PdfReader

        try:
            reader = reader_factory(str(file_path))
            text_parts = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n".join(part for part in text_parts if part)
            return ParsedText(
                file_type="pdf",
                text=text,
                metadata={"page_count": len(reader.pages)},
            )
        except Exception as exc:  # pragma: no cover - defensive parser boundary
            raise ParsingError(f"failed to parse PDF: {file_path}") from exc
