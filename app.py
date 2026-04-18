"""
FastAPI application entry point.

Routes:
  POST /upload              — Upload tài liệu, build RAGIndex (FAISS + BM25 data)
  POST /ask                 — Hỏi đáp với search_mode tùy chọn (vector hoặc hybrid)
  POST /compare             — So sánh kết quả vector vs hybrid trên cùng câu hỏi
  DELETE /vectorstore       — Xóa toàn bộ index trong memory
  DELETE /vectorstore/{session_id}/{file_id} — Xóa index một file trong phiên
"""
try:
    # Load biến môi trường từ file .env khi chạy local (py app.py)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Nếu không có python-dotenv hoặc không cần .env, bỏ qua
    pass

from collections import defaultdict
from contextlib import asynccontextmanager
from langchain_core.documents.base import Document
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
from langchain_core.documents import Document
from datetime import datetime
from src.parsers.pdf_parser import extract_documents_pdf
from src.parsers.docx_parser import extract_documents_docx
from src.rag.pipeline import (
    build_index,
    ask_question,
    compare_search_modes,
    merge_rag_indices,
)
from src.history.router import router as history_router
from src.history.router import append_message, get_recent_messages
from src.database import get_pool, close_pool, db_insert_session, db_insert_document
from src.models import RAGIndex, AskRequest, AskResponse, CompareResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo DB connection pool khi app start, đóng khi shutdown."""
    await get_pool()
    yield
    await close_pool()


app = FastAPI(
    lifespan=lifespan,
    title="PTPMNM RAG API",
    description=(
        "RAG API hỗ trợ citation tracking và hybrid search (BM25 + vector).\n\n"
        "**Chế độ retrieval:**\n"
        "- `vector`: FAISS semantic search — tốt với câu hỏi ngữ nghĩa\n"
        "- `hybrid`: BM25 + FAISS qua EnsembleRetriever — tốt với cả keyword lẫn ngữ nghĩa\n\n"
        "Dùng `/compare` để đánh giá hiệu quả của từng chế độ."
    ),
    version="3.0.0",
)
app.include_router(history_router)

# ── In-memory store: session_id → { file_id → RAGIndex } ───────────────────
# Lưu ý: dữ liệu mất khi restart server. Cần persistent storage cho production.
INDEX_DB: dict[str, dict[str, RAGIndex]] = defaultdict(dict)


def _merged_index_for_files(
    session_id: str, file_ids: list[str]
) -> tuple[RAGIndex | None, str | None]:
    """
    Trả về (RAGIndex đã gộp, None) hoặc (None, message lỗi).
    """
    by_file = INDEX_DB[session_id]
    indices: list[RAGIndex] = []
    for fid in file_ids:
        if fid not in by_file:
            return None, f"file_id không thuộc session này: {fid}"
        indices.append(by_file[fid])
    try:
        return merge_rag_indices(indices), None
    except ValueError as exc:
        return None, str(exc)


# ─────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────

def load_documents(file_path: Path) -> list[Document]:
    """
    Bóc tách tài liệu PDF hoặc DOCX thành list Document có metadata.

    Args:
        file_path: Đường dẫn file đã lưu trên disk.

    Returns:
        List Document kèm metadata (page/paragraph/source).

    Raises:
        ValueError: Nếu định dạng file không hỗ trợ.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_documents_pdf(file_path)
    elif suffix == ".docx":
        return extract_documents_docx(file_path)
    raise ValueError(f"Định dạng file không hỗ trợ: {file_path.suffix}")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/upload", summary="Upload tài liệu PDF/DOCX để indexing")
async def upload(files: List[UploadFile] = File(...)):
    """
    Upload file PDF hoặc DOCX, bóc tách nội dung và xây dựng RAGIndex.

    RAGIndex bao gồm:
    - FAISS vector store (dùng cho vector/hybrid search)
    - Danh sách chunk thuần text (dùng cho BM25 index lúc query)

    Returns:
        `session_id`, `files` (mỗi phần tử: `file_id`, `filename`, `document_count`)
        — client gửi `session_id` + `file_ids` (list) cho `/ask` và `/compare`.
    """
    session_id = str(uuid.uuid4())
    await db_insert_session(session_id)

    uploaded: list[dict] = []

    for file in files:
        file_id = str(uuid.uuid4())
        file_path = Path(f"data/{file_id}_{file.filename}")
        file_path.parent.mkdir(exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        documents = load_documents(file_path)
        index = build_index(documents)

        # print("Đây là chunk", index.chunks)
        # print("Đây là vector", index.vectorstore)

        
        suffix = file_path.suffix.lower().lstrip(".") or "unknown"
        await db_insert_document(
            file_id,
            session_id,
            file.filename or "unnamed",
            suffix,
        )

        # Lưu FAISS index trong memory theo session/file
        INDEX_DB[session_id][file_id] = index
        uploaded.append(
            {
                "file_id": file_id,
                "filename": file.filename,
                "document_count": len(documents),
            }
        )

    return {
        "message": "Upload thành công",
        "session_id": session_id,
        "files": uploaded,
    }


@app.post("/ask", response_model=AskResponse, summary="Hỏi đáp với tài liệu đã upload")
async def ask(req: AskRequest):
    """
    Hỏi đáp về tài liệu. Gửi `file_ids`: một hoặc nhiều id trong session —
    nhiều id thì retrieval gộp chunk của các file đó. Chọn search_mode để điều khiển retrieval:

    - **hybrid** (mặc định): kết hợp BM25 keyword search + FAISS semantic search.
      Tốt với hầu hết câu hỏi, đặc biệt câu hỏi chứa thuật ngữ/tên riêng chính xác.

    - **vector**: chỉ dùng FAISS semantic search.
      Tốt với câu hỏi mang tính diễn đạt lại, paraphrase.

    Trường `bm25_weight` điều chỉnh tỷ lệ ảnh hưởng của BM25 trong hybrid mode.
    """
    if req.session_id not in INDEX_DB:
        return AskResponse(
            question=req.question,
            answer="Lỗi: Session không tồn tại. Vui lòng upload tài liệu trước.",
            citations=[],
            search_mode=req.search_mode,
        )


    index, err = _merged_index_for_files(req.session_id, req.file_ids)
    if err:
        return AskResponse(
            question=req.question,
            answer=f"Lỗi: {err}",
            citations=[],
            search_mode=req.search_mode,
        )

    chat_history = await get_recent_messages(req.session_id)
    result = ask_question(
        index=index,
        question=req.question,
        chat_history=chat_history,
        search_mode=req.search_mode,
        bm25_weight=req.bm25_weight,
    )
    await append_message(req.session_id, req.question, result.answer)
    return result


@app.post(
    "/compare",
    response_model=CompareResponse,
    summary="So sánh vector search vs hybrid search trên cùng một câu hỏi",
    tags=["Evaluation"],
)
async def compare(req: AskRequest):
    """
    Chạy cùng một câu hỏi trên cả hai chế độ retrieval và trả về kết quả song song.

    **Mục đích:** Đánh giá chất lượng retrieval của từng chế độ. Cho phép developer
    và người dùng thấy trực tiếp sự khác nhau giữa vector search và hybrid search.

    **Lưu ý:** Endpoint này gọi LLM 2 lần nên latency cao hơn `/ask` thông thường.
    Chỉ dùng cho mục đích đánh giá, không dùng trong production flow.

    Response kèm `latency_ms` của từng mode để so sánh performance.
    """
    if req.session_id not in INDEX_DB:
        raise HTTPException(
            status_code=404,
            detail="session_id không tồn tại. Vui lòng upload tài liệu trước.",
        )

    index, err = _merged_index_for_files(req.session_id, req.file_ids)
    if err:
        raise HTTPException(status_code=404, detail=err)

    chat_history = await get_recent_messages(req.session_id)

    vector_result, hybrid_result = compare_search_modes(
        index=index,
        question=req.question,
        chat_history=chat_history,
        bm25_weight=req.bm25_weight,
    )

    return CompareResponse(
        question=req.question,
        vector_result=vector_result,
        hybrid_result=hybrid_result,
    )


@app.delete(
    "/vectorstore",
    tags=["Vector Store"],
    summary="Xóa toàn bộ index trong memory",
)
async def clear_all_vectorstore():
    """Xóa toàn bộ RAGIndex (vectorstore + chunks) của tất cả file trong memory."""
    count = len(INDEX_DB)
    INDEX_DB.clear()
    return {"message": f"Đã xóa toàn bộ {count} tài liệu khỏi memory"}


@app.delete(
    "/vectorstore/{session_id}/{file_id}",
    tags=["Vector Store"],
    summary="Xóa index của một file trong một phiên",
)
async def clear_vectorstore(session_id: str, file_id: str):
    """Xóa RAGIndex của một file trong session."""
    by_file = INDEX_DB.get(session_id)
    if not by_file or file_id not in by_file:
        return {"error": "session_id hoặc file_id không tồn tại trong memory"}
    del by_file[file_id]
    if not by_file:
        del INDEX_DB[session_id]
    return {"message": f"Đã xóa index file {file_id} trong session {session_id}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
