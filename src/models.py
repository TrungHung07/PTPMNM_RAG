"""
Shared data models cho tính năng Citation/Source Tracking.

Module này định nghĩa các Pydantic model dùng chung giữa RAG pipeline
và FastAPI response schemas để đảm bảo tính nhất quán của dữ liệu.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class CitationSource(BaseModel):
    """
    Mô tả một đoạn văn bản gốc được dùng làm nguồn tham chiếu (citation)
    khi AI tạo ra câu trả lời.

    Attributes:
        content: Nội dung đoạn văn bản gốc được truy xuất từ tài liệu.
        metadata: Thông tin vị trí của đoạn văn, bao gồm:
                  - 'source': tên file gốc (ví dụ: 'report.pdf')
                  - 'page': số trang (chỉ có với file PDF)
                  - 'paragraph': số thứ tự đoạn văn (chỉ có với file DOCX)
                  - 'chunk_index': số thứ tự chunk trong trang/đoạn văn đó
    """
    content: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    """
    Schema phản hồi đầy đủ của endpoint POST /ask,
    bao gồm câu trả lời từ AI và danh sách nguồn trích dẫn.

    Attributes:
        question: Câu hỏi người dùng đã gửi.
        answer: Câu trả lời do AI sinh ra dựa trên tài liệu.
        citations: Danh sách các đoạn văn gốc được dùng làm cơ sở trả lời.
                   Mỗi item chứa content và metadata (vị trí trong tài liệu).
    """
    question: str
    answer: str
    citations: list[CitationSource]


__all__ = ["CitationSource", "AskResponse"]
