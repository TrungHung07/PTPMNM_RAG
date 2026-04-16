from __future__ import annotations

from pathlib import Path
from typing import Literal

PdfBackend = Literal["pdfplumber", "pypdf"]


def extract_text_pdfplumber(path: Path) -> str:
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text_parts: list[str] = []
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_pdf(path: Path, backend: PdfBackend = "pdfplumber") -> str:
    if backend == "pdfplumber":
        return extract_text_pdfplumber(path)
    if backend == "pypdf":
        return extract_text_pypdf(path)
    raise ValueError(f"Unsupported PDF backend: {backend}")


__all__ = ["extract_text_pdf", "extract_text_pdfplumber", "extract_text_pypdf"]
