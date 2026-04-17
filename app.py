from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid
from pydantic import BaseModel

from src.parsers.pdf_parser import extract_text_pdf
from src.parsers.docx_parser import extract_text_docx
from src.rag.pipeline import build_vectorstore_from_text, ask_question
from src.history.router import router as history_router
from src.history.router import init_history, append_message, get_recent_messages
from src.database import get_pool, close_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()
    yield
    await close_pool()


app = FastAPI(lifespan=lifespan)
app.include_router(history_router)

VECTOR_DB = {}


class AskRequest(BaseModel):
    file_id: str
    question: str


# =========================
# LOAD FILE
# =========================
def load_text(file_path: Path) -> str:
    if file_path.suffix == ".pdf":
        return extract_text_pdf(file_path)
    elif file_path.suffix == ".docx":
        return extract_text_docx(file_path)
    else:
        raise ValueError("Unsupported file")


# =========================
# API
# =========================

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = Path(f"data/{file_id}_{file.filename}")

    file_path.parent.mkdir(exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_text(file_path)
    vectorstore = build_vectorstore_from_text(text)

    VECTOR_DB[file_id] = vectorstore
    await init_history(file_id, file.filename)

    return {
        "message": "upload thành công",
        "file_id": file_id
    }


@app.post("/ask")
async def ask(req: AskRequest):
    if req.file_id not in VECTOR_DB:
        return {"error": "file_id không tồn tại"}

    vectorstore = VECTOR_DB[req.file_id]

    chat_history = await get_recent_messages(req.file_id)
    answer = ask_question(vectorstore, req.question, chat_history=chat_history)

    await append_message(req.file_id, req.question, answer)

    return {
        "question": req.question,
        "answer": answer
    }


@app.delete("/vectorstore", tags=["Vector Store"], summary="Xóa toàn bộ tài liệu đã upload")
async def clear_all_vectorstore():
    """Xóa toàn bộ vector store của tất cả các file."""
    VECTOR_DB.clear()
    return {"message": "Đã xóa toàn bộ tài liệu"}


@app.delete("/vectorstore/{file_id}", tags=["Vector Store"], summary="Xóa tài liệu của một file")
async def clear_vectorstore(file_id: str):
    """Xóa vector store của một file cụ thể."""
    if file_id not in VECTOR_DB:
        return {"error": "file_id không tồn tại"}
    del VECTOR_DB[file_id]
    return {"message": f"Đã xóa tài liệu của file {file_id}"}
