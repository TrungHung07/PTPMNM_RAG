from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid
from pydantic import BaseModel

from src.parsers.pdf_parser import extract_documents_pdf
from src.parsers.docx_parser import extract_documents_docx
from src.rag.pipeline import build_vectorstore_from_documents, ask_question
from src.history.router import router as history_router
from src.history.router import init_history, append_message, get_recent_messages
from src.database import get_pool, close_pool
from src.models import AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo connection pool khi app start, đóng khi app shutdown."""
    await get_pool()
    yield
    await close_pool()


app = FastAPI(
    lifespan=lifespan,
    title="PTPMNM RAG API",
    description="RAG API hỗ trợ trích dẫn nguồn gốc (citation tracking) từ tài liệu PDF/DOCX.",
    version="2.0.0",
)
app.include_router(history_router)

# Vector store in-memory: {file_id: vectorstore}
# Lưu ý: dữ liệu sẽ mất khi restart server, cần persistent storage cho production
VECTOR_DB = {}


class AskRequest(BaseModel):
    """Request body cho endpoint POST /ask."""
    file_id: str
    question: str


# =========================
# LOAD FILE
# =========================

def load_documents(file_path: Path):
    """
    Bóc tách tài liệu từ file PDF hoặc DOCX thành danh sách Document có metadata.

    Hỗ trợ hai định dạng:
    - .pdf: mỗi Document tương ứng một trang, metadata chứa 'page' và 'source'.
    - .docx: mỗi Document tương ứng một đoạn văn, metadata chứa 'paragraph' và 'source'.

    Args:
        file_path: Đường dẫn đầy đủ tới file cần đọc.

    Returns:
        Danh sách Document kèm metadata, dùng để xây dựng vector store.

    Raises:
        ValueError: Nếu định dạng file không được hỗ trợ.
    """
    if file_path.suffix.lower() == ".pdf":
        return extract_documents_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return extract_documents_docx(file_path)
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {file_path.suffix}")


# =========================
# API ENDPOINTS
# =========================

@app.post("/upload", summary="Upload tài liệu PDF/DOCX để indexing")
async def upload(file: UploadFile = File(...)):
    """
    Upload file PDF hoặc DOCX, bóc tách nội dung, tạo vector store và lưu vào memory.

    Quá trình xử lý:
    1. Lưu file tạm vào thư mục data/
    2. Bóc tách text kèm metadata (trang/đoạn văn)
    3. Chunk, embed và tạo FAISS vector store
    4. Khởi tạo lịch sử chat trong database

    Returns:
        JSON chứa 'file_id' dùng để query trong các request tiếp theo.
    """
    file_id = str(uuid.uuid4())
    file_path = Path(f"data/{file_id}_{file.filename}")
    file_path.parent.mkdir(exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Bóc tách tài liệu thành Documents có metadata (page/paragraph/source)
    documents = load_documents(file_path)
    vectorstore = build_vectorstore_from_documents(documents)

    VECTOR_DB[file_id] = vectorstore
    await init_history(file_id, file.filename)

    return {
        "message": "Upload thành công",
        "file_id": file_id,
        "document_count": len(documents),   # Số trang/đoạn văn đã được index
    }


@app.post("/ask", response_model=AskResponse, summary="Đặt câu hỏi về tài liệu đã upload")
async def ask(req: AskRequest):
    """
    Hỏi đáp dựa trên tài liệu đã upload, trả về câu trả lời kèm danh sách citations.

    Mỗi citation trong response chứa:
    - content: đoạn văn bản gốc từ tài liệu mà AI đã dùng để trả lời
    - metadata.source: tên file gốc
    - metadata.page: số trang (chỉ với PDF)
    - metadata.paragraph: số đoạn văn (chỉ với DOCX)
    - metadata.chunk_index: số thứ tự chunk trong trang/đoạn đó

    Args:
        req: AskRequest gồm file_id và question.

    Returns:
        AskResponse chứa question, answer và danh sách citations.
    """
    if req.file_id not in VECTOR_DB:
        return AskResponse(
            question=req.question,
            answer="Lỗi: file_id không tồn tại. Vui lòng upload tài liệu trước.",
            citations=[],
        )

    vectorstore = VECTOR_DB[req.file_id]

    # Lấy lịch sử hội thoại gần nhất từ database
    chat_history = await get_recent_messages(req.file_id)

    # Gọi RAG pipeline — trả về AskResponse có đầy đủ citations
    result = ask_question(vectorstore, req.question, chat_history=chat_history)

    # Lưu câu hỏi và câu trả lời vào lịch sử
    await append_message(req.file_id, req.question, result.answer)

    return result


@app.delete("/vectorstore", tags=["Vector Store"], summary="Xóa toàn bộ tài liệu đã upload")
async def clear_all_vectorstore():
    """Xóa toàn bộ vector store của tất cả các file trong memory."""
    count = len(VECTOR_DB)
    VECTOR_DB.clear()
    return {"message": f"Đã xóa toàn bộ {count} tài liệu"}


@app.delete("/vectorstore/{file_id}", tags=["Vector Store"], summary="Xóa tài liệu của một file")
async def clear_vectorstore(file_id: str):
    """Xóa vector store của một file cụ thể theo file_id."""
    if file_id not in VECTOR_DB:
        return {"error": "file_id không tồn tại"}
    del VECTOR_DB[file_id]
    return {"message": f"Đã xóa tài liệu của file {file_id}"}
