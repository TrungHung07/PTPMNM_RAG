"""
FastAPI application entry point.

Routes:
  POST /upload                        — Upload tài liệu, build RAGIndex (FAISS + BM25 data)
  POST /ask                           — Hỏi đáp với search_mode tùy chọn (vector hoặc hybrid)
  POST /compare                       — So sánh kết quả vector vs hybrid trên cùng câu hỏi
  POST /sessions/{session_id}/restore — Restore RAG index từ disk cho một phiên cụ thể
  DELETE /vectorstore                 — Xóa toàn bộ index trong memory
  DELETE /vectorstore/{session_id}/{file_id} — Xóa index một file trong phiên

Cơ chế persist:
  - DB (PostgreSQL): lưu session / documents / messages — không bao giờ mất
  - data/           : lưu file gốc upload — persist qua restart
  - INDEX_DB        : FAISS index in-memory — tự động rebuilt từ data/ khi cần
"""
try:
    # Load biến môi trường từ file .env khi chạy local (py app.py)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Nếu không có python-dotenv hoặc không cần .env, bỏ qua
    pass

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

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
    run_rag,
)
from src.graph_rag import build_graph_index, merge_graph_indices, run_graph_rag
from src.graph_rag.types import GraphRAGIndex as GraphRAGIndexType
from src.rag.llm import get_llm
from src.history.router import router as history_router
from src.history.router import append_message, get_recent_messages
from src.database import get_pool, close_pool, db_insert_session, db_insert_document, db_get_documents_by_session
from src.models import RAGIndex, AskRequest, AskResponse, CompareResponse, CompareRAGResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động (Warm-up) các model để tránh latency lượt đầu
    print("🚀 Đang khởi tạo các AI models (Embedding & Reranker)...")
    try:
        from src.rag.rerank import get_reranker
        from src.rag.embedding import get_embedding
        
        # Load Reranker
        get_reranker()
        # Load Embedding
        get_embedding()
        
        # Load LLM (Ollama) - Chạy thử một prompt cực ngắn để nạp model
        from src.rag.llm import get_llm
        llm = get_llm()
        llm.invoke("Hi") 
        
        print("✅ Các AI models đã sẵn sàng!")
    except Exception as e:
        print(f"⚠️ Lỗi khi khởi tạo models: {e}")

    """Khởi tạo DB connection pool khi app start, đóng khi shutdown."""
    await get_pool()
    yield
    await close_pool()
    print("👋 Đang đóng ứng dụng...")


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

# ── In-memory store: session_id → { file_id → RAGIndex } ────────────────────────
# Lưu ý: dữ liệu mất khi restart server. Cần persistent storage cho production.
INDEX_DB: dict[str, dict[str, RAGIndex]] = defaultdict(dict)

# ── In-memory store cho Graph RAG: session_id → { file_id → GraphRAGIndex } ──
# Song song với INDEX_DB. Được rebuild cùng lúc với Standard RAG.
GRAPH_DB: dict[str, dict[str, GraphRAGIndexType]] = defaultdict(dict)


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


def _merged_graph_for_files(
    session_id: str, file_ids: list[str]
) -> tuple[GraphRAGIndexType | None, str | None]:
    """
    Trả về (GraphRAGIndex đã gộp, None) hoặc (None, message lỗi).
    Dung sai: nếu session chưa có trong GRAPH_DB (do restart trước khi có Graph RAG)
    thì trả lỗi rõ để user upload lại.
    """
    by_file = GRAPH_DB.get(session_id, {})
    indices: list[GraphRAGIndexType] = []
    for fid in file_ids:
        if fid not in by_file:
            return None, (
                f"Graph index chưa có cho file {fid}. "
                f"Vui lòng upload lại để build Graph RAG index."
            )
        indices.append(by_file[fid])
    try:
        return merge_graph_indices(indices), None
    except ValueError as exc:
        return None, str(exc)


async def _ensure_graph_indices(session_id: str, file_ids: list[str]) -> str | None:
    """
    Đảm bảo GRAPH_DB có index cho tất cả file_ids trong session.

    - Nếu session chưa có Standard index trong memory → auto-restore Standard từ disk.
    - Nếu file nào chưa có Graph index → build on-demand từ RAGIndex.chunks.

    Returns:
        None nếu ok, hoặc message lỗi (dạng string) để trả ra API response.
    """
    # 1) Ensure Standard index exists (restore nếu cần)
    if session_id not in INDEX_DB:
        missing = await _restore_session_from_disk(session_id, restore_graph=False)
        if "__session_not_found__" in missing:
            return "Session không tồn tại. Vui lòng upload tài liệu trước."
        if missing:
            return f"Không tìm thấy file trên disk: {', '.join(missing)}. Vui lòng upload lại."

    # 2) Build Graph index on-demand cho các file thiếu
    by_std = INDEX_DB.get(session_id, {})
    by_graph = GRAPH_DB.get(session_id, {})
    llm = get_llm()

    for fid in file_ids:
        if fid in by_graph:
            continue
        if fid not in by_std:
            return f"file_id không thuộc session này: {fid}"
        try:
            graph_index = build_graph_index(by_std[fid].chunks, llm=llm)
        except Exception as exc:
            return f"Lỗi build Graph index cho file_id={fid}: {exc}"
        GRAPH_DB[session_id][fid] = graph_index

    return None


# ─────────────────────────────────────────────
# Internal helpers
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
    elif suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": file_path.name})]
    raise ValueError(f"Định dạng file không hỗ trợ: {file_path.suffix}")


async def _restore_session_from_disk(
    session_id: str,
    restore_graph: bool = False,
) -> list[str]:
    """
    Tìm file gốc trên disk và rebuild FAISS index cho tất cả documents
    thuộc session_id. Cập nhật INDEX_DB nếu thành công.

    File trên disk được lưu theo pattern: data/{doc_id}_{file_name}

    Args:
        session_id: ID phiên cần restore.
        restore_graph: Nếu True thì rebuild thêm Graph RAG index (chậm).
                       Mặc định False — chỉ rebuild Standard RAG index.

    Returns:
        Danh sách tên file bị thiếu (không tìm thấy trên disk).
        List rỗng nghĩa là restore thành công toàn bộ.
    """
    docs = await db_get_documents_by_session(session_id)
    if not docs:
        return ["__session_not_found__"]

    missing: list[str] = []
    for doc in docs:
        file_id   = str(doc["doc_id"])
        file_name = doc["file_name"]

        # Tìm file theo prefix doc_id (phần tên gốc có thể chứa ký tự đặc biệt)
        data_dir  = Path("data")
        candidates = list(data_dir.glob(f"{file_id}_*"))
        if not candidates:
            missing.append(file_name)
            continue

        file_path = candidates[0]
        try:
            documents = load_documents(file_path)
            index     = build_index(documents)
            INDEX_DB[session_id][file_id] = index
            if restore_graph:
                try:
                    graph_index = build_graph_index(index.chunks, llm=get_llm())
                    GRAPH_DB[session_id][file_id] = graph_index
                except Exception as g_exc:
                    print(f"⚠️ Graph RAG restore lỗi cho {file_name}: {g_exc}")
        except Exception as exc:
            missing.append(f"{file_name} ({exc})")

    return missing


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/upload", summary="Upload tài liệu PDF/DOCX để indexing (Standard RAG)")
async def upload(
    files: List[UploadFile] = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Upload file PDF hoặc DOCX, bóc tách nội dung và xây dựng RAGIndex (Standard).

    Chỉ build FAISS vectorstore + chunk list cho Standard RAG (vector/hybrid search).
    **Không** build Graph RAG index — dùng `/upload-graph` nếu cần Graph RAG.

    Args:
        files: Danh sách file cần upload.
        chunk_size: Kích thước mỗi chunk (mặc định 1000).
        chunk_overlap: Độ chồng lấp giữa các chunk (mặc định 200).

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
        index = build_index(documents, chunk_size=chunk_size, overlap=chunk_overlap)

        suffix = file_path.suffix.lower().lstrip(".") or "unknown"
        await db_insert_document(
            file_id,
            session_id,
            file.filename or "unnamed",
            suffix,
        )

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

    **Auto-restore:** Nếu server vừa restart, RAG index được tự động rebuild từ
    file gốc trên disk — người dùng không cần upload lại.
    """
    # Nếu session chưa có trong memory → thử auto-restore từ disk
    if req.session_id not in INDEX_DB:
        missing = await _restore_session_from_disk(req.session_id)

        if "__session_not_found__" in missing:
            return AskResponse(
                question=req.question,
                answer="Lỗi: Session không tồn tại. Vui lòng upload tài liệu trước.",
                citations=[],
                search_mode=req.search_mode,
            )

        if missing:
            # Một số file bị thiếu trên disk
            return AskResponse(
                question=req.question,
                answer=(
                    f"Lỗi: Không tìm thấy file trên disk: {', '.join(missing)}.\n"
                    f"Vui lòng upload lại các file bị thiếu để tiếp tục."
                ),
                citations=[],
                search_mode=req.search_mode,
            )
        # Restore thành công — tiếp tục bình thường

    chat_history = await get_recent_messages(req.session_id, req.file_ids)

    # ── Standard RAG path (default) ──────────────────────────────────────────
    index, err = _merged_index_for_files(req.session_id, req.file_ids)
    if err:
        return AskResponse(
            question=req.question,
            answer=f"Lỗi: {err}",
            citations=[],
            search_mode=req.search_mode,
        )

    result = ask_question(
        index=index,
        question=req.question,
        chat_history=chat_history,
        search_mode=req.search_mode,
        bm25_weight=req.bm25_weight,
        rerank_enabled=req.rerank_enabled,
        rerank_threshold=req.rerank_threshold,
    )
    await append_message(
        req.session_id,
        req.question,
        result.answer,
        req.file_ids,
        search_mode=req.search_mode,
        citations=[c.model_dump() for c in result.citations],
    )
    return result


@app.post("/ask-graph", response_model=AskResponse, summary="Hỏi đáp Graph RAG (build graph on-demand)")
async def ask_graph(req: AskRequest):
    """
    Graph RAG endpoint tách riêng khỏi `/ask`.

    Hành vi:
    - Đảm bảo Standard index tồn tại (auto-restore nếu server restart).
    - Nếu Graph index chưa có cho file_ids → build on-demand từ chunks (LLM extraction).
    - Sau đó chạy Graph RAG retrieval + LLM để trả lời.
    """
    err = await _ensure_graph_indices(req.session_id, req.file_ids)
    if err:
        return AskResponse(
            question=req.question,
            answer=f"Lỗi Graph RAG: {err}",
            citations=[],
            search_mode="graph",
        )

    chat_history = await get_recent_messages(req.session_id, req.file_ids)
    graph_index, g_err = _merged_graph_for_files(req.session_id, req.file_ids)
    if g_err:
        return AskResponse(
            question=req.question,
            answer=f"Lỗi Graph RAG: {g_err}",
            citations=[],
            search_mode="graph",
        )

    result = run_graph_rag(
        graph_index=graph_index,
        question=req.question,
        chat_history=chat_history,
        llm=get_llm(),
    )
    await append_message(
        req.session_id,
        req.question,
        result.answer,
        req.file_ids,
        search_mode="graph",
        citations=[c.model_dump() for c in result.citations],
    )
    return AskResponse(
        question=req.question,
        answer=result.answer,
        citations=result.citations,
        search_mode="graph",
    )



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
        missing = await _restore_session_from_disk(req.session_id)
        if "__session_not_found__" in missing:
            raise HTTPException(
                status_code=404,
                detail="session_id không tồn tại. Vui lòng upload tài liệu trước.",
            )
        if missing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Không tìm thấy file trên disk: {', '.join(missing)}. "
                    f"Vui lòng upload lại các file bị thiếu để tiếp tục."
                ),
            )

    index, err = _merged_index_for_files(req.session_id, req.file_ids)
    if err:
        raise HTTPException(status_code=404, detail=err)

    chat_history = await get_recent_messages(req.session_id, req.file_ids)

    vector_result, hybrid_result = compare_search_modes(
        index=index,
        question=req.question,
        chat_history=chat_history,
        bm25_weight=req.bm25_weight,
    )
    
    # Lưu 2 messages riêng biệt: vector trước, hybrid sau
    await append_message(
        req.session_id,
        req.question,
        vector_result.answer,
        req.file_ids,
        search_mode="compare_vector",
        citations=[c.model_dump() for c in vector_result.citations],
    )
    await append_message(
        req.session_id,
        req.question,
        hybrid_result.answer,
        req.file_ids,
        search_mode="compare_hybrid",
        citations=[c.model_dump() for c in hybrid_result.citations],
    )

    return CompareResponse(
        question=req.question,
        vector_result=vector_result,
        hybrid_result=hybrid_result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# So sánh Standard RAG vs Graph RAG (parallel ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

# Thread pool khởi tạo 1 lần — tránh overhead tạo pool mỗi request
_COMPARE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="compare_rag")


@app.post(
    "/compare-rag",
    response_model=CompareRAGResponse,
    summary="So sánh Standard RAG vs Graph RAG trên cùng câu hỏi (song song)",
    tags=["Evaluation"],
)
async def compare_rag_endpoint(req: AskRequest):
    """
    Chạy cùng một câu hỏi trên Standard RAG (hybrid) VÀ Graph RAG đồng thời
    bằng ThreadPoolExecutor, trả kết quả song song 2 cột.

    **Parallelism:** Standard RAG retrieval (FAISS+BM25) và Graph RAG retrieval
    (graph lookup) chạy trong 2 thread riêng. Cả 2 đều gọi LLM nên Ollama sẽ
    queue, nhưng overhead retrieval được overlap → tiết kiệm thời gian đáng kể
    so với sequential (đặc biệt khi graph lookup + entity extraction chậm).

    **History:** Lưu 2 message riêng biệt vào DB:
    - search_mode="compare_standard" cho Standard RAG
    - search_mode="compare_graph" cho Graph RAG
    """
    # ── Restore nếu mất index do restart ─────────────────────────────────────
    if req.session_id not in INDEX_DB:
        missing = await _restore_session_from_disk(req.session_id)
        if "__session_not_found__" in missing:
            raise HTTPException(status_code=404, detail="session_id không tồn tại.")
        if missing:
            raise HTTPException(
                status_code=409,
                detail=f"Không tìm thấy file trên disk: {', '.join(missing)}.",
            )

    # ── Merge index ───────────────────────────────────────────────────────────
    std_index, std_err = _merged_index_for_files(req.session_id, req.file_ids)
    if std_err:
        raise HTTPException(status_code=404, detail=std_err)

    grp_index, grp_err = _merged_graph_for_files(req.session_id, req.file_ids)
    if grp_err:
        raise HTTPException(
            status_code=422,
            detail=f"Graph RAG: {grp_err}. Vui lòng upload lại tài liệu để build Graph index.",
        )

    chat_history = await get_recent_messages(req.session_id, req.file_ids)
    llm = get_llm()

    # ── Định nghĩa 2 hàm sync để chạy trong thread ───────────────────────────
    def _run_standard() -> SearchResult:
        return run_rag(
            index=std_index,
            question=req.question,
            chat_history=chat_history,
            search_mode="hybrid",
            bm25_weight=req.bm25_weight,
            rerank_enabled=req.rerank_enabled,
            rerank_threshold=req.rerank_threshold,
        )

    def _run_graph() -> SearchResult:
        return run_graph_rag(
            graph_index=grp_index,
            question=req.question,
            chat_history=chat_history,
            llm=llm,
        )

    # ── Submit song song, await cả 2 ─────────────────────────────────────────
    loop = asyncio.get_event_loop()
    std_future = loop.run_in_executor(_COMPARE_EXECUTOR, _run_standard)
    grp_future = loop.run_in_executor(_COMPARE_EXECUTOR, _run_graph)
    standard_result, graph_result = await asyncio.gather(std_future, grp_future)

    # ── Lưu history 2 messages ────────────────────────────────────────────────
    await append_message(
        req.session_id, req.question, standard_result.answer, req.file_ids,
        search_mode="compare_standard",
        citations=[c.model_dump() for c in standard_result.citations],
    )
    await append_message(
        req.session_id, req.question, graph_result.answer, req.file_ids,
        search_mode="compare_graph",
        citations=[c.model_dump() for c in graph_result.citations],
    )

    return CompareRAGResponse(
        question=req.question,
        standard_result=standard_result,
        graph_result=graph_result,
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


@app.post(
    "/sessions/{session_id}/restore",
    tags=["Session"],
    summary="Restore RAG index từ disk cho một phiên cụ thể",
)
async def restore_session(session_id: str):
    """
    Rebuild FAISS index cho một phiên từ file gốc đã lưu trên disk.

    Endpoint này thường **không cần gọi thủ công** vì `/ask` và `/compare`
    đã tự động restore khi phát hiện session chưa có trong memory.

    Dùng khi:
    - Muốn pre-warm cache trước khi user bắt đầu chat
    - Debug / kiểm tra xem session có thể restore được không

    Returns:
        `restored_files`: danh sách file đã rebuild thành công.
        `missing_files` : danh sách file không tìm thấy trên disk (cần upload lại).
    """
    if session_id in INDEX_DB:
        docs = await db_get_documents_by_session(session_id)
        return {
            "message": "Session đã có trong memory, không cần restore.",
            "session_id": session_id,
            "restored_files": [d["file_name"] for d in docs],
            "missing_files": [],
        }

    missing = await _restore_session_from_disk(session_id)

    if "__session_not_found__" in missing:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{session_id}' không tồn tại trong database.",
        )

    docs = await db_get_documents_by_session(session_id)
    restored = [d["file_name"] for d in docs if d["file_name"] not in missing]

    return {
        "message": "Restore hoàn tất." if not missing else "Restore một phần — một số file bị thiếu.",
        "session_id": session_id,
        "restored_files": restored,
        "missing_files": missing,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
