from __future__ import annotations

from pathlib import Path


def extract_text_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    paragraphs: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


__all__ = ["extract_text_docx"]
