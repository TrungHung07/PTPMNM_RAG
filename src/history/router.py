# ============================================================
# Module: Chat History
# Người phụ trách: [Tên thành viên phụ trách phần này]
#
# Chức năng:
#   - Lưu trữ lịch sử hội thoại theo từng file (session)
#   - API lấy lịch sử để hiển thị trên sidebar
# ============================================================

from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

router = APIRouter(prefix="/history", tags=["Chat History"])

# ── In-memory store ──────────────────────────────────────────
# Key   : file_id (str)
# Value : { "filename": str, "messages": [{question, answer}] }
CHAT_HISTORY: dict[str, dict] = {}


# ── Pydantic models ──────────────────────────────────────────
class ChatMessage(BaseModel):
    question: str
    answer: str


class SessionSummary(BaseModel):
    file_id: str
    filename: str
    message_count: int


class HistoryResponse(BaseModel):
    file_id: str
    filename: str
    history: List[ChatMessage]


# ── Helper functions (dùng nội bộ, gọi từ app.py) ────────────
def init_history(file_id: str, filename: str) -> None:
    """Khởi tạo lịch sử rỗng cho một file mới upload."""
    CHAT_HISTORY[file_id] = {"filename": filename, "messages": []}


def append_message(file_id: str, question: str, answer: str) -> None:
    """Lưu một cặp hỏi-đáp vào lịch sử của file."""
    if file_id not in CHAT_HISTORY:
        CHAT_HISTORY[file_id] = {"filename": "unknown", "messages": []}
    CHAT_HISTORY[file_id]["messages"].append({"question": question, "answer": answer})


# ── API endpoints ─────────────────────────────────────────────
@router.get("", response_model=List[SessionSummary])
async def get_all_sessions():
    """Trả về danh sách tóm tắt tất cả các phiên chat (dùng cho sidebar)."""
    return [
        {
            "file_id": file_id,
            "filename": session["filename"],
            "message_count": len(session["messages"]),
        }
        for file_id, session in CHAT_HISTORY.items()
    ]


@router.get("/{file_id}", response_model=HistoryResponse)
async def get_history(file_id: str):
    """Trả về toàn bộ lịch sử hội thoại của một file."""
    if file_id not in CHAT_HISTORY:
        return {"file_id": file_id, "filename": "", "history": []}
    session = CHAT_HISTORY[file_id]
    return {
        "file_id": file_id,
        "filename": session["filename"],
        "history": session["messages"],
    }


@router.delete("", summary="Xóa toàn bộ lịch sử chat")
async def clear_all_history():
    """Xóa toàn bộ lịch sử hội thoại của tất cả các file."""
    CHAT_HISTORY.clear()
    return {"message": "Đã xóa toàn bộ lịch sử chat"}


@router.delete("/{file_id}", summary="Xóa lịch sử chat của một file")
async def clear_history(file_id: str):
    """Xóa lịch sử hội thoại của một file cụ thể."""
    if file_id not in CHAT_HISTORY:
        return {"error": "file_id không tồn tại"}
    del CHAT_HISTORY[file_id]
    return {"message": f"Đã xóa lịch sử chat của file {file_id}"}
