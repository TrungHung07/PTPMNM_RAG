"""
src/rag/pipeline.py — RAG Pipeline với hỗ trợ Hybrid Search.

Orchestrate toàn bộ luồng từ tài liệu → vector/BM25 index → retrieval → LLM → response.
Hỗ trợ 2 chế độ retrieval:
  - "vector": chỉ dùng FAISS semantic search (hành vi cũ)
  - "hybrid": kết hợp BM25 + FAISS qua EnsembleRetriever (mặc định mới)
"""
from __future__ import annotations

import os
import logging
import time
from typing import Literal

from langchain_core.documents import Document

from src.chunking.text_chunker import chunk_documents
from src.rag.embedding import get_embedding
from src.rag.vectorstore import create_vectorstore
from src.rag.retriever import build_vector_retriever, build_hybrid_retriever
from src.rag.llm import get_llm
from src.rag.rerank import rerank_documents
from src.models import RAGIndex, AskResponse, CitationSource, SearchResult

# Số lượng tin nhắn gần nhất đưa vào ngữ cảnh hội thoại
HISTORY_WINDOW = 5

# Số chunk mỗi retriever trả về trước khi fusion (EnsembleRetriever dùng giá trị này)
RETRIEVER_TOP_K = 4


_logger = logging.getLogger("uvicorn.error")

_NO_INFO_PHRASE = "Không tìm thấy thông tin trong tài liệu."


# ─────────────────────────────────────────────
# Index Building
# ─────────────────────────────────────────────

def build_index(
    documents: list[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> RAGIndex:
    """
    Xây dựng RAGIndex (FAISS vectorstore + chunk list) từ danh sách Document có metadata.

    Đây là hàm entry point cho việc xử lý tài liệu mới. Kết quả trả về là
    RAGIndex chứa ĐỦ dữ liệu để chạy cả vector search lẫn BM25 search.

    Quy trình:
    1. Chia Documents thành chunk nhỏ (giữ nguyên metadata page/paragraph/source).
    2. Embed các chunk bằng HuggingFace sentence-transformers → FAISS index.
    3. Lưu lại danh sách chunk (plain text) để xây BM25 index lúc query.

    Args:
        documents: List[Document] từ pdf_parser hoặc docx_parser (có metadata).
        chunk_size: Số ký tự tối đa mỗi chunk (mặc định: 1000).
        overlap: Số ký tự overlap giữa các chunk liên tiếp (mặc định: 200).

    Returns:
        RAGIndex với .vectorstore (FAISS) và .chunks (list[Document]).
    """
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    embedding = get_embedding()
    vectorstore = create_vectorstore(chunks, embedding)
    return RAGIndex(vectorstore=vectorstore, chunks=chunks)


def build_vectorstore_from_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
):
    """
    [Legacy] Wrapper giữ tương thích ngược — trả về chỉ vectorstore (không có chunks).

    Các code cũ gọi hàm này sẽ chỉ dùng được vector search, không dùng được hybrid.
    Hãy migrate sang build_index() để hỗ trợ đầy đủ hybrid search.

    Args:
        documents: List[Document] có metadata.
        chunk_size: Số ký tự tối đa mỗi chunk.
        overlap: Số ký tự overlap.

    Returns:
        FAISS vector store (không kèm chunks).
    """
    return build_index(documents, chunk_size, overlap).vectorstore


def build_vectorstore_from_text(text: str):
    """
    [Legacy] Xây dựng FAISS vector store từ chuỗi văn bản thuần túy.
    Không có metadata, không hỗ trợ citation hay hybrid search.

    Args:
        text: Chuỗi văn bản đầy đủ.

    Returns:
        FAISS vector store.
    """
    doc = Document(page_content=text, metadata={"source": "unknown"})
    return build_index([doc]).vectorstore


def merge_rag_indices(indices: list[RAGIndex]) -> RAGIndex:
    """
    Gộp nhiều RAGIndex (từng file trong session) thành một index để hỏi đáp một lần.

    Ghép list chunk rồi embed lại toàn bộ vào một FAISS — không mutate index gốc trong
    INDEX_DB. Chi phí tăng khi danh sách file dài / nhiều chunk; có thể tối ưu sau bằng
    merge FAISS inplace + cache.
    """
    if not indices:
        raise ValueError("merge_rag_indices: indices rỗng")
    if len(indices) == 1:
        return indices[0]
    chunks: list[Document] = []
    for ri in indices:
        chunks.extend(ri.chunks)
    embedding = get_embedding()
    vectorstore = create_vectorstore(chunks, embedding)
    return RAGIndex(vectorstore=vectorstore, chunks=chunks)


# ─────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────

def _format_chat_history(chat_history: list) -> str:
    """
    Chuyển list [{question, answer}] thành chuỗi hội thoại cho prompt.

    Args:
        chat_history: List dict {'question': str, 'answer': str}.

    Returns:
        Chuỗi "Human: ...\nAI: ..." hoặc "" nếu rỗng.
    """
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        lines.append(f"Human: {msg['question']}")
        lines.append(f"AI: {msg['answer']}")
    return "\n".join(lines)




def _citations_match_answer(answer: str, citations: list[CitationSource]) -> list[CitationSource]:
    """Nếu model báo không có trong tài liệu thì không trả citation lệch từ retrieval."""
    if not citations:
        return citations
    text = answer or ""
    if _NO_INFO_PHRASE in text or "Không tìm thấy thông tin trong tài liệu" in text:
        return []
    return citations


def _build_citation_list(docs: list[Document]) -> list[CitationSource]:
    """
    Chuyển list Document được retriever trả về thành list CitationSource.

    Args:
        docs: List Document từ kết quả retrieval.

    Returns:
        List CitationSource dùng trong API response.
    """
    return [
        CitationSource(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]


def _build_prompt(context: str, question: str, history_text: str) -> str:
    """
    Xây dựng prompt cho LLM từ context, lịch sử và câu hỏi.

    Tách hàm này ra riêng để dễ test nội dung prompt mà không cần call LLM.

    Args:
        context: Nội dung các chunk đã retrieve, ghép thành chuỗi.
        question: Câu hỏi của người dùng.
        history_text: Lịch sử hội thoại đã được format.

    Returns:
        Chuỗi prompt hoàn chỉnh sẵn sàng gửi cho LLM.
    """
    history_section = f"""
Lịch sử hội thoại trước đó (để hiểu ngữ cảnh follow-up):
{history_text}
""" if history_text else ""

    return f"""
Bạn là AI chỉ được phép trả lời dựa trên thông tin trong tài liệu dưới đây.

QUY TẮC:
- CHỈ sử dụng thông tin có trong Context.
- KHÔNG được suy đoán, KHÔNG được dùng kiến thức phổ thông / Wikipedia / định nghĩa có sẵn nếu không có trong Context.
- Nếu Context không chứa thông tin để trả lời trực tiếp câu hỏi, CHỈ trả lời đúng một câu:
  "Không tìm thấy thông tin trong tài liệu." — không giải thích thêm.
- Nếu câu hỏi là follow-up (ví dụ: "giải thích thêm", "ý đó là gì"), hãy dùng Lịch sử hội thoại để hiểu ngữ cảnh (vẫn chỉ được dùng nội dung có trong Context cho phần trả lời).
- Nếu không tìm thấy câu trả lời trong Context, hãy trả lời:
  "Không tìm thấy thông tin trong tài liệu."

Context từ tài liệu:
{context}
{history_section}
Câu hỏi hiện tại: {question}
""".strip()


# ─────────────────────────────────────────────
# Core Q&A Functions
# ─────────────────────────────────────────────

def run_rag(
    index: RAGIndex,
    question: str,
    chat_history: list,
    search_mode: Literal["vector", "hybrid"] = "hybrid",
    bm25_weight: float = 0.5,
) -> SearchResult:
    """
    Thực thi một lần RAG query, đo latency và trả về SearchResult.

    Đây là hàm cốt lõi được dùng bởi cả ask_question() và compare_search_modes().
    Tách riêng để:
    - ask_question() gọi 1 lần với mode mong muốn
    - compare_search_modes() gọi 2 lần (vector + hybrid) và so sánh

    Args:
        index: RAGIndex chứa vectorstore và chunks đã xây dựng lúc upload.
        question: Câu hỏi của người dùng.
        chat_history: Lịch sử hội thoại [{question, answer}].
        search_mode: "hybrid" (mặc định) hoặc "vector".
        bm25_weight: Trọng số BM25 khi dùng hybrid mode [0.0, 1.0].

    Returns:
        SearchResult chứa answer, citations và latency_ms.
    """
    t_start = time.perf_counter()

    rerank_enabled = os.getenv("RERANK_ENABLED", "false").strip().lower() == "true"
    rerank_debug = os.getenv("RERANK_DEBUG", "false").strip().lower() == "true"
    retrieve_candidates = int(os.getenv("RETRIEVE_CANDIDATES", str(RETRIEVER_TOP_K)))
    context_top_k = int(os.getenv("RERANK_TOP_K", str(RETRIEVER_TOP_K)))
    rerank_max_chars = int(os.getenv("RERANK_MAX_CHARS", "2000"))

    if rerank_debug:
        _logger.warning(
            "[RAG] config: rerank_enabled=%s rerank_debug=%s retrieve_candidates=%s context_top_k=%s rerank_max_chars=%s",
            rerank_enabled,
            rerank_debug,
            retrieve_candidates,
            context_top_k,
            rerank_max_chars,
        )
    # ── Chọn retriever phù hợp với mode ──────────────────────────────────────
    t_retrieval_start = time.perf_counter()
    if search_mode == "hybrid":
        retriever = build_hybrid_retriever(
            vectorstore=index.vectorstore,
            chunks=index.chunks,
            k=retrieve_candidates,
            bm25_weight=bm25_weight,
        )
    else:
        retriever = build_vector_retriever(index.vectorstore, k=retrieve_candidates)

    # ── Retrieve docs (sau đó lọc theo relevance để khớp citation ↔ câu hỏi) ──
    docs = retriever.invoke(question)

    
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000
    if rerank_debug:
        _logger.warning(
            "[RAG] retrieval done: mode=%s candidates=%s/%s retrieval_ms=%.2f",
            search_mode,
            len(docs),
            retrieve_candidates,
            retrieval_ms,
        )

    # Không gọi LLM khi không có chunk đạt ngưỡng — tránh model dùng kiến thức có sẵn
    if not docs:
        latency_ms = (time.perf_counter() - t_start) * 1000
        return SearchResult(
            answer=_NO_INFO_PHRASE,
            citations=[],
            latency_ms=round(latency_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=0.0,
            llm_ms=0.0,
            rerank_enabled=rerank_enabled,
            retrieve_candidates=retrieve_candidates,
            context_top_k=context_top_k,
        )

    # ── Optional re-ranking (cross-encoder) ──────────────────────────────────
    t_rerank_start = time.perf_counter()
    if rerank_enabled:
        docs, _scores = rerank_documents(
            question,
            docs,
            top_k=context_top_k,
            max_chars=rerank_max_chars,
            enabled=True,
        )
    else:
        docs = docs[: max(0, context_top_k)]
    rerank_ms = (time.perf_counter() - t_rerank_start) * 1000
    if rerank_debug:
        _logger.warning(
            "[RAG] rerank: enabled=%s kept=%s/%s rerank_ms=%.2f",
            rerank_enabled,
            len(docs),
            context_top_k,
            rerank_ms,
        )

    # ── Build prompt và gọi LLM ───────────────────────────────────────────────
    context = "\n\n".join(doc.page_content for doc in docs)
    recent_history = chat_history[-HISTORY_WINDOW:]
    history_text = _format_chat_history(recent_history)
    prompt = _build_prompt(context, question, history_text)

    llm = get_llm()
    t_llm_start = time.perf_counter()
    answer = llm.invoke(prompt)
    llm_ms = (time.perf_counter() - t_llm_start) * 1000
    if rerank_debug:
        total_ms = (time.perf_counter() - t_start) * 1000
        _logger.warning(
            "[RAG] llm_ms=%.2f total_ms=%.2f (retrieval_ms=%.2f, rerank_ms=%.2f)",
            llm_ms,
            total_ms,
            retrieval_ms,
            rerank_ms,
        )

    citations = _build_citation_list(docs)
    citations = _citations_match_answer(answer, citations)

    latency_ms = (time.perf_counter() - t_start) * 1000

    return SearchResult(
        answer=answer,
        citations=citations,
        latency_ms=round(latency_ms, 2),
        retrieval_ms=round(retrieval_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        llm_ms=round(llm_ms, 2),
        rerank_enabled=rerank_enabled,
        retrieve_candidates=retrieve_candidates,
        context_top_k=context_top_k,
    )


def ask_question(
    index: RAGIndex,
    question: str,
    chat_history: list = [],
    search_mode: Literal["vector", "hybrid"] = "hybrid",
    bm25_weight: float = 0.5,
) -> AskResponse:
    """
    API chính để hỏi đáp về tài liệu với lựa chọn chế độ retrieval.

    Wrapper mỏng gọi run_rag() và đóng gói kết quả vào AskResponse.

    Args:
        index: RAGIndex của tài liệu đã upload.
        question: Câu hỏi của người dùng.
        chat_history: Lịch sử hội thoại gần đây.
        search_mode: "hybrid" (mặc định) hoặc "vector".
        bm25_weight: Trọng số BM25 cho hybrid mode [0.0, 1.0].

    Returns:
        AskResponse chứa question, answer, citations và search_mode đã dùng.
    """
    result = run_rag(index, question, chat_history, search_mode, bm25_weight)
    return AskResponse(
        question=question,
        answer=result.answer,
        citations=result.citations,
        search_mode=search_mode,
    )


def compare_search_modes(
    index: RAGIndex,
    question: str,
    chat_history: list = [],
    bm25_weight: float = 0.5,
) -> tuple[SearchResult, SearchResult]:
    """
    Chạy cùng một câu hỏi trên cả hai chế độ retrieval và trả về kết quả để so sánh.

    Hàm này intentionally gọi LLM 2 lần — chỉ dùng cho mục đích đánh giá,
    không nên dùng trong production flow thông thường.

    Thứ tự: vector trước, hybrid sau (để latency đo được không bị ảnh hưởng
    bởi cache warm-up của embedding model).

    Args:
        index: RAGIndex chứa vectorstore và chunks.
        question: Câu hỏi cần so sánh.
        chat_history: Lịch sử hội thoại.
        bm25_weight: Trọng số BM25 cho hybrid mode.

    Returns:
        tuple (vector_result, hybrid_result) — cả hai đều là SearchResult.
    """
    vector_result = run_rag(index, question, chat_history, "vector", bm25_weight)
    hybrid_result = run_rag(index, question, chat_history, "hybrid", bm25_weight)
    return vector_result, hybrid_result


__all__ = [
    "build_index",
    "build_vectorstore_from_documents",
    "build_vectorstore_from_text",
    "merge_rag_indices",
    "ask_question",
    "run_rag",
    "compare_search_modes",
]