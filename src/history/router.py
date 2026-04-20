# ============================================================
# Module: Chat History
# Người phụ trách: [Tên thành viên phụ trách phần này]
#
# Chức năng:
#   - Lưu trữ lịch sử hội thoại theo từng phiên (session) vào PostgreSQL
#   - Một session có thể chứa nhiều file (multi-file upload)
#   - API lấy lịch sử để hiển thị trên sidebar
# ============================================================

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from src.database import (
    db_get_all_sessions,
    db_get_documents_by_session,
    db_delete_session,
    db_delete_all_sessions,
    db_append_message,
    db_get_messages,
    db_get_recent_messages,
)

router = APIRouter(prefix="/history", tags=["Chat History"])


# ── Pydantic models ──────────────────────────────────────────

class ChatMessage(BaseModel):
    question: str
    answer: str
    created_at: Optional[datetime] = None
    file_ids: List[str] = []   # doc_id của các file đã dùng để trả lời lượt này


class FileInfo(BaseModel):
    """Thông tin một file trong phiên."""
    file_id: str       # doc_id (UUID)
    file_name: str
    file_type: str
    uploaded_at: Optional[datetime] = None


class SessionSummary(BaseModel):
    """Tóm tắt một phiên — dùng để render sidebar."""
    session_id: str
    created_at: Optional[datetime] = None
    files: List[FileInfo]          # danh sách tất cả file trong phiên
    file_count: int
    message_count: int


class SessionHistoryResponse(BaseModel):
    """Chi tiết lịch sử hội thoại của một phiên."""
    session_id: str
    created_at: Optional[datetime] = None
    files: List[FileInfo]          # danh sách tất cả file trong phiên
    history: List[ChatMessage]


# ── Helper functions (gọi từ app.py) ─────────────────────────

async def append_message(
    session_id: str,
    question: str,
    answer: str,
    file_ids: list[str],
) -> None:
    """Lưu một cặp hỏi-đáp vào DB kèm danh sách file đã dùng để trả lời."""
    await db_append_message(session_id, question, answer, file_ids)


async def get_recent_messages(
    session_id: str,
    file_ids: list[str],
    limit: int = 5,
) -> list:
    """Lấy N tin nhắn gần nhất đã dùng cùng tập file — dùng cho Conversational RAG."""
    return await db_get_recent_messages(session_id, file_ids, limit)



# ── API endpoints ─────────────────────────────────────────────

@router.get("", response_model=List[SessionSummary], summary="Lấy danh sách tất cả phiên chat")
async def get_all_sessions():
    """
    Trả về danh sách tóm tắt tất cả các phiên chat (dùng cho sidebar).

    Mỗi phiên bao gồm:
    - `session_id`: ID phiên
    - `files`: danh sách tất cả file đã upload trong phiên (file_id, file_name, file_type)
    - `file_count`: tổng số file trong phiên
    - `message_count`: tổng số tin nhắn trong phiên
    """
    rows = await db_get_all_sessions()
    result = []
    for r in rows:
        # Lấy danh sách files đầy đủ từ bảng documents
        docs = await db_get_documents_by_session(str(r["session_id"]))
        result.append(
            SessionSummary(
                session_id=str(r["session_id"]),
                created_at=r.get("created_at"),
                files=[
                    FileInfo(
                        file_id=str(d["doc_id"]),
                        file_name=d["file_name"],
                        file_type=d["file_type"],
                        uploaded_at=d.get("uploaded_at"),
                    )
                    for d in docs
                ],
                file_count=int(r["file_count"]),
                message_count=int(r["message_count"]),
            )
        )
    return result


@router.get("/{session_id}", response_model=SessionHistoryResponse, summary="Lấy lịch sử hội thoại của một phiên")
async def get_session_history(session_id: str):
    """
    Trả về toàn bộ lịch sử hội thoại của một phiên chat.

    Bao gồm:
    - Danh sách tất cả file đã upload trong phiên
    - Toàn bộ cặp hỏi-đáp theo thứ tự thời gian
    """
    # Lấy danh sách documents trong session
    docs = await db_get_documents_by_session(session_id)
    # Không cần session tồn tại tường minh — documents trống cũng hợp lệ
    # nhưng nếu không có docs VÀ không có messages thì 404
    messages = await db_get_messages(session_id)

    if not docs and not messages:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{session_id}' không tồn tại hoặc không có dữ liệu."
        )

    files = [
        FileInfo(
            file_id=str(d["doc_id"]),
            file_name=d["file_name"],
            file_type=d["file_type"],
            uploaded_at=d.get("uploaded_at"),
        )
        for d in docs
    ]

    history = [
        ChatMessage(
            question=m["question"],
            answer=m["answer"],
            created_at=m.get("created_at"),
            file_ids=m.get("file_ids") or [],
        )
        for m in messages
    ]

    return SessionHistoryResponse(
        session_id=session_id,
        files=files,
        history=history,
    )


@router.delete("", summary="Xóa toàn bộ lịch sử chat")
async def clear_all_history():
    """Xóa toàn bộ phiên chat, documents và messages liên quan (CASCADE)."""
    await db_delete_all_sessions()
    return {"message": "Đã xóa toàn bộ lịch sử chat"}


@router.delete("/{session_id}", summary="Xóa một phiên chat cụ thể")
async def clear_session_history(session_id: str):
    """
    Xóa một phiên chat theo `session_id`.
    Do ràng buộc CASCADE trong DB, toàn bộ documents và messages của phiên này cũng bị xóa.
    """
    deleted = await db_delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{session_id}' không tồn tại."
        )
    return {"message": f"Đã xóa phiên chat {session_id}"}
