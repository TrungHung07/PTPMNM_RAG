"""
Public API của package src.parsers.

Export cả hàm legacy (extract_text_*) và hàm mới hỗ trợ citation (extract_documents_*).
"""
from .docx_parser import extract_text_docx, extract_documents_docx
from .pdf_parser import (
    extract_text_pdf,
    extract_documents_pdf,
    extract_documents_pdfplumber,
    extract_documents_pypdf,
)

__all__ = [
    # Legacy (tương thích ngược)
    "extract_text_docx",
    "extract_text_pdf",
    # Mới - hỗ trợ citation tracking
    "extract_documents_docx",
    "extract_documents_pdf",
    "extract_documents_pdfplumber",
    "extract_documents_pypdf",
]
