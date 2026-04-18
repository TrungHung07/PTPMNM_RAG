"""
Shared data models cho RAG API.

Module này định nghĩa các Pydantic model dùng chung giữa:
  - src/rag/pipeline.py  (RAGIndex, AskResponse)
  - app.py              (AskRequest, CompareResponse)

Việc tập trung schema tại đây giúp:
  - API layer và Business logic cùng dùng một kiểu dữ liệu
  - Dễ generate OpenAPI docs nhờ Pydantic annotations
  - Dễ unittest schema validation độc lập
"""
from __future__ import annotations

from typing import Any, Literal
from dataclasses import dataclass, field

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# RAG Core Types (dùng nội bộ trong pipeline)
# ─────────────────────────────────────────────

@dataclass
class RAGIndex:
    """
    Container lưu trữ kết quả indexing của một tài liệu.

    Lưu cả vectorstore lẫn danh sách chunk (words) để:
    - vectorstore: dùng cho semantic search (FAISS)
    - chunks: dùng để xây dựng BM25 index lúc query (không cần lưu riêng)

    Kế thừa từ dataclass thay vì Pydantic để tránh serialize
    các object phức tạp (FAISS, Document list) không cần thiết.

    Attributes:
        vectorstore: FAISS vector store đã được index.
        chunks: Danh sách Document đã chia chunk, dùng cho BM25Retriever.
    """
    vectorstore: Any                      # FAISS object (không type hint cụ thể tránh circular)
    chunks: list[Document] = field(default_factory=list)


# ─────────────────────────────────────────────
# API Request / Response Schemas (Pydantic)
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    """
    Request body cho endpoint POST /ask và POST /compare.

    Attributes:
        session_id: Phiên upload (trả về từ POST /upload).
        file_ids: Danh sách file trong phiên cần truy xuất — 1 id = một file,
                  nhiều id = gộp context retrieval trên các file đó (theo thứ tự).
        question: Câu hỏi.
        search_mode: Chiến lược tìm kiếm (/ask): "hybrid" hoặc "vector".
        bm25_weight: Trọng số BM25 trong hybrid mode [0.0, 1.0].
    """
    session_id: str
    file_ids: list[str] = Field(..., min_length=1)
    question: str
    search_mode: Literal["vector", "hybrid"] = Field(
        default="hybrid",
        description="Chiến lược retrieval: 'hybrid' (BM25+vector) hoặc 'vector' (chỉ semantic)"
    )
    bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Trọng số BM25 trong hybrid mode [0.0, 1.0]. Vector weight = 1 - giá trị này."
    )

    @field_validator("file_ids", mode="after")
    @classmethod
    def normalize_file_ids(cls, v: list[str]) -> list[str]:
        cleaned = [x.strip() for x in v if x.strip()]
        if not cleaned:
            raise ValueError("file_ids phải có ít nhất một id hợp lệ")
        return list(dict.fromkeys(cleaned))


class CitationSource(BaseModel):
    """
    Mô tả một đoạn văn bản gốc được dùng làm nguồn tham chiếu (citation)
    khi AI tạo ra câu trả lời.

    Attributes:
        content: Nội dung đoạn văn bản gốc được truy xuất từ tài liệu.
        metadata: Thông tin vị trí:
                  - 'source': tên file gốc (vd: 'report.pdf')
                  - 'page': số trang (chỉ với PDF)
                  - 'paragraph': số thứ tự đoạn (chỉ với DOCX)
                  - 'chunk_index': số thứ tự chunk trong trang/đoạn đó
    """
    content: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    """
    Response đầy đủ của endpoint POST /ask.

    Attributes:
        question: Câu hỏi người dùng đã gửi.
        answer: Câu trả lời do AI sinh ra dựa trên nội dung tài liệu.
        citations: Các đoạn văn gốc được dùng làm căn cứ trả lời.
        search_mode: Chế độ tìm kiếm đã được sử dụng ("vector" hoặc "hybrid").
    """
    question: str
    answer: str
    citations: list[CitationSource]
    search_mode: str = "vector"     # Ghi lại chế độ đã dùng để client biết


class SearchResult(BaseModel):
    """
    Kết quả của một lần thực thi RAG (answer + citations + metadata hiệu năng).
    Dùng làm sub-object trong CompareResponse.

    Attributes:
        answer: Câu trả lời AI sinh ra.
        citations: Danh sách nguồn trích dẫn.
        latency_ms: Thời gian xử lý tổng cộng tính bằng millisecond.
    """
    answer: str
    citations: list[CitationSource]
    latency_ms: float = Field(description="Thời gian xử lý (ms)")


class CompareResponse(BaseModel):
    """
    Response của endpoint POST /compare — so sánh trực tiếp hai chế độ retrieval.

    Endpoint này chạy cùng một câu hỏi trên cả hai chế độ và trả về kết quả
    song song để người dùng/developer đánh giá chất lượng retrieval.

    Lưu ý: Endpoint này gọi LLM 2 lần, nên chỉ dùng để đánh giá, không production.

    Attributes:
        question: Câu hỏi gốc.
        vector_result: Kết quả từ pure vector search.
        hybrid_result: Kết quả từ hybrid search (BM25 + vector).
    """
    question: str
    vector_result: SearchResult
    hybrid_result: SearchResult


__all__ = [
    "RAGIndex",
    "AskRequest",
    "CitationSource",
    "AskResponse",
    "SearchResult",
    "CompareResponse",
]
