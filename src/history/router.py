# ============================================================
# Module: Chat History
# Người phụ trách: [Tên thành viên phụ trách phần này]
#
# Chức năng:
#   - Lưu trữ lịch sử hội thoại theo từng file (session) vào PostgreSQL
#   - API lấy lịch sử để hiển thị trên sidebar
# ============================================================

from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

from src.database import (
    db_get_all_sessions,
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


class SessionSummary(BaseModel):
    file_id: str  # session UUID (giữ tên field cho client cũ)
    filename: str
    message_count: int


class HistoryResponse(BaseModel):
    file_id: str
    filename: str
    history: List[ChatMessage]


# ── Helper functions (gọi từ app.py) ─────────────────────────
async def append_message(session_id: str, question: str, answer: str) -> None:
    """Lưu một cặp hỏi-đáp vào DB (theo phiên chat)."""
    await db_append_message(session_id, question, answer)


async def get_recent_messages(session_id: str, limit: int = 5) -> list:
    """Lấy N tin nhắn gần nhất - dùng cho Conversational RAG."""
    return await db_get_recent_messages(session_id, limit)


# ── API endpoints ─────────────────────────────────────────────
@router.get("", response_model=List[SessionSummary])
async def get_all_sessions():
    """Trả về danh sách tóm tắt tất cả các phiên chat (dùng cho sidebar)."""
    rows = await db_get_all_sessions()
    return [
        {
            "file_id": str(r["session_id"]),
            "filename": r["filename"] or "",
            "message_count": r["message_count"],
        }
        for r in rows
    ]



@router.get("/{file_id}", response_model=HistoryResponse)
async def get_history(file_id: str):
    """Trả về toàn bộ lịch sử hội thoại của một file."""
    sessions = await db_get_all_sessions()
    session = next((s for s in sessions if str(s["session_id"]) == file_id), None)
    if not session:
        return {"file_id": file_id, "filename": "", "history": []}

    messages = await db_get_messages(file_id)
    return {
        "file_id": file_id,
        "filename": session["filename"],
        "history": messages,
    }


@router.delete("", summary="Xóa toàn bộ lịch sử chat")
async def clear_all_history():
    """Xóa toàn bộ lịch sử hội thoại của tất cả các file."""
    await db_delete_all_sessions()
    return {"message": "Đã xóa toàn bộ lịch sử chat"}


@router.delete("/{file_id}", summary="Xóa lịch sử chat của một file")
async def clear_history(file_id: str):
    """Xóa lịch sử hội thoại của một file cụ thể."""
    deleted = await db_delete_session(file_id)
    if not deleted:
        return {"error": "file_id không tồn tại"}
    return {"message": f"Đã xóa lịch sử chat của file {file_id}"}
