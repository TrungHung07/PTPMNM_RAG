"""
FastAPI application entry point.

Routes:
  POST /upload              — Upload tài liệu, build RAGIndex (FAISS + BM25 data)
  POST /ask                 — Hỏi đáp với search_mode tùy chọn (vector hoặc hybrid)
  POST /compare             — So sánh kết quả vector vs hybrid trên cùng câu hỏi
  DELETE /vectorstore       — Xóa toàn bộ index trong memory
  DELETE /vectorstore/{id}  — Xóa index của một file cụ thể
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid
from langchain_core.documents import Document

from src.parsers.pdf_parser import extract_documents_pdf
from src.parsers.docx_parser import extract_documents_docx
from src.rag.pipeline import build_index, ask_question, compare_search_modes
from src.history.router import router as history_router
from src.history.router import init_history, append_message, get_recent_messages
from src.database import get_pool, close_pool
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

# ── In-memory store: file_id → RAGIndex (vectorstore + chunks) ──────────────
# Lưu ý: dữ liệu mất khi restart server. Cần persistent storage cho production.
INDEX_DB: dict[str, RAGIndex] = {}


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
async def upload(file: UploadFile = File(...)):
    """
    Upload file PDF hoặc DOCX, bóc tách nội dung và xây dựng RAGIndex.

    RAGIndex bao gồm:
    - FAISS vector store (dùng cho vector/hybrid search)
    - Danh sách chunk thuần text (dùng cho BM25 index lúc query)

    Returns:
        JSON chứa file_id (dùng cho các request tiếp theo) và số trang/đoạn đã index.
    """
    file_id = str(uuid.uuid4())
    file_path = Path(f"data/{file_id}_{file.filename}")
    file_path.parent.mkdir(exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Bóc tách → build index (FAISS + lưu chunks cho BM25)
    documents = load_documents(file_path)
    index = build_index(documents)

    INDEX_DB[file_id] = index
    await init_history(file_id, file.filename)

    return {
        "message": "Upload thành công",
        "file_id": file_id,
        "document_count": len(documents),
        "chunk_count": len(index.chunks),
    }


@app.post("/ask", response_model=AskResponse, summary="Hỏi đáp với tài liệu đã upload")
async def ask(req: AskRequest):
    """
    Hỏi đáp về tài liệu. Chọn search_mode để điều khiển chiến lược retrieval:

    - **hybrid** (mặc định): kết hợp BM25 keyword search + FAISS semantic search.
      Tốt với hầu hết câu hỏi, đặc biệt câu hỏi chứa thuật ngữ/tên riêng chính xác.

    - **vector**: chỉ dùng FAISS semantic search.
      Tốt với câu hỏi mang tính diễn đạt lại, paraphrase.

    Trường `bm25_weight` điều chỉnh tỷ lệ ảnh hưởng của BM25 trong hybrid mode.
    """
    if req.file_id not in INDEX_DB:
        return AskResponse(
            question=req.question,
            answer="Lỗi: file_id không tồn tại. Vui lòng upload tài liệu trước.",
            citations=[],
            search_mode=req.search_mode,
        )

    index = INDEX_DB[req.file_id]
    chat_history = await get_recent_messages(req.file_id)

    result = ask_question(
        index=index,
        question=req.question,
        chat_history=chat_history,
        search_mode=req.search_mode,
        bm25_weight=req.bm25_weight,
    )

    await append_message(req.file_id, req.question, result.answer)
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
    if req.file_id not in INDEX_DB:
        return {"error": "file_id không tồn tại. Vui lòng upload tài liệu trước."}

    index = INDEX_DB[req.file_id]
    chat_history = await get_recent_messages(req.file_id)

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
    "/vectorstore/{file_id}",
    tags=["Vector Store"],
    summary="Xóa index của một file cụ thể",
)
async def clear_vectorstore(file_id: str):
    """Xóa RAGIndex của một file theo file_id."""
    if file_id not in INDEX_DB:
        return {"error": "file_id không tồn tại"}
    del INDEX_DB[file_id]
    return {"message": f"Đã xóa index của file {file_id}"}
