from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.documents import Document

PdfBackend = Literal["pdfplumber", "pypdf"]


def extract_documents_pdfplumber(path: Path) -> list[Document]:
    """
    Bóc tách nội dung từng trang của file PDF bằng thư viện pdfplumber.

    Mỗi trang PDF được chuyển thành một đối tượng Document chứa:
    - page_content: nội dung văn bản của trang
    - metadata: chứa 'page' (số trang, 1-indexed) và 'source' (tên file)

    Args:
        path: Đường dẫn tới file PDF cần đọc.

    Returns:
        Danh sách Document, mỗi phần tử tương ứng với một trang PDF có nội dung.
    """
    import pdfplumber

    documents: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                documents.append(Document(
                    page_content=page_text,
                    metadata={
                        "page": i + 1,          # Số trang, bắt đầu từ 1
                        "source": path.name,    # Tên file dùng để hiển thị citation
                    }
                ))
    return documents


def extract_documents_pypdf(path: Path) -> list[Document]:
    """
    Bóc tách nội dung từng trang của file PDF bằng thư viện pypdf.

    Mỗi trang PDF được chuyển thành một đối tượng Document chứa:
    - page_content: nội dung văn bản của trang
    - metadata: chứa 'page' (số trang, 1-indexed) và 'source' (tên file)

    Args:
        path: Đường dẫn tới file PDF cần đọc.

    Returns:
        Danh sách Document, mỗi phần tử tương ứng với một trang PDF có nội dung.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for i, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            documents.append(Document(
                page_content=page_text,
                metadata={
                    "page": i + 1,          # Số trang, bắt đầu từ 1
                    "source": path.name,    # Tên file dùng để hiển thị citation
                }
            ))
    return documents


def extract_text_pdf(path: Path, backend: PdfBackend = "pdfplumber") -> str:
    """
    [Legacy] Bóc tách toàn bộ text từ file PDF thành một chuỗi duy nhất.
    Hàm này giữ lại để tương thích ngược. Với tính năng citation, dùng
    extract_documents_pdf() thay thế.

    Args:
        path: Đường dẫn tới file PDF.
        backend: Thư viện đọc PDF, 'pdfplumber' (mặc định) hoặc 'pypdf'.

    Returns:
        Toàn bộ nội dung file PDF dưới dạng một chuỗi văn bản.
    """
    docs = extract_documents_pdf(path, backend)
    return "\n\n".join(doc.page_content for doc in docs)


def extract_documents_pdf(path: Path, backend: PdfBackend = "pdfplumber") -> list[Document]:
    """
    Bóc tách nội dung file PDF thành danh sách Document (kèm metadata page/source).

    Args:
        path: Đường dẫn tới file PDF cần đọc.
        backend: Thư viện đọc PDF: 'pdfplumber' (mặc định) hoặc 'pypdf'.

    Returns:
        Danh sách Document có metadata, mỗi phần tử tương ứng một trang PDF.

    Raises:
        ValueError: Nếu backend không hợp lệ.
    """
    if backend == "pdfplumber":
        return extract_documents_pdfplumber(path)
    if backend == "pypdf":
        return extract_documents_pypdf(path)
    raise ValueError(f"Unsupported PDF backend: {backend}")


__all__ = [
    "extract_text_pdf",
    "extract_documents_pdf",
    "extract_documents_pdfplumber",
    "extract_documents_pypdf",
]
