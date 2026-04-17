from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import shutil
import uuid
from pydantic import BaseModel

from src.chunking.text_chunker import chunk_text
from src.parsers.pdf_parser import extract_text_pdf
from src.parsers.docx_parser import extract_text_docx

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.llms import Ollama

from src.rag.pipeline import build_vectorstore_from_text, ask_question

app = FastAPI()

VECTOR_DB = {}

class AskRequest(BaseModel):
    file_id: str
    question: str
    
# =========================
# LOAD FILE (reuse code bạn)
# =========================
def load_text(file_path: Path) -> str:
    if file_path.suffix == ".pdf":
        return extract_text_pdf(file_path)
    elif file_path.suffix == ".docx":
        return extract_text_docx(file_path)
    else:
        raise ValueError("Unsupported file")


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

    return {
        "message": "upload thành công",
        "file_id": file_id
    }


@app.post("/ask")
async def ask(req: AskRequest):
        if req.file_id not in VECTOR_DB:
            return {"error": "file_id không tồn tại"}

        vectorstore = VECTOR_DB[req.file_id]

        answer = ask_question(vectorstore, req.question)

        return {
            "question": req.question,
            "answer": answer
        }

