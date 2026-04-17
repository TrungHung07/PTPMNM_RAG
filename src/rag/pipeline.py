"""
src/rag/pipeline.py — RAG Pipeline với hỗ trợ Hybrid Search.

Orchestrate toàn bộ luồng từ tài liệu → vector/BM25 index → retrieval → LLM → response.
Hỗ trợ 2 chế độ retrieval:
  - "vector": chỉ dùng FAISS semantic search (hành vi cũ)
  - "hybrid": kết hợp BM25 + FAISS qua EnsembleRetriever (mặc định mới)
"""
from __future__ import annotations

import time
from typing import Literal

from langchain_core.documents import Document

from src.chunking.text_chunker import chunk_documents
from src.rag.embedding import get_embedding
from src.rag.vectorstore import create_vectorstore
from src.rag.retriever import build_vector_retriever, build_hybrid_retriever
from src.rag.llm import get_llm
from src.models import RAGIndex, CitationSource, AskResponse, SearchResult

# Số lượng tin nhắn gần nhất đưa vào ngữ cảnh hội thoại
HISTORY_WINDOW = 5

# Số chunk mỗi retriever trả về trước khi fusion (EnsembleRetriever dùng giá trị này)
RETRIEVER_TOP_K = 4


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
- KHÔNG được suy đoán, KHÔNG thêm kiến thức bên ngoài.
- Nếu câu hỏi là follow-up (ví dụ: "giải thích thêm", "ý đó là gì"), hãy dùng Lịch sử hội thoại để hiểu ngữ cảnh.
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

    # ── Chọn retriever phù hợp với mode ──────────────────────────────────────
    if search_mode == "hybrid":
        retriever = build_hybrid_retriever(
            vectorstore=index.vectorstore,
            chunks=index.chunks,
            k=RETRIEVER_TOP_K,
            bm25_weight=bm25_weight,
        )
    else:
        retriever = build_vector_retriever(index.vectorstore, k=RETRIEVER_TOP_K)

    # ── Retrieve docs (cũng chính là nguồn citation) ─────────────────────────
    docs = retriever.invoke(question)

    # ── Build prompt và gọi LLM ───────────────────────────────────────────────
    context = "\n\n".join(doc.page_content for doc in docs)
    recent_history = chat_history[-HISTORY_WINDOW:]
    history_text = _format_chat_history(recent_history)
    prompt = _build_prompt(context, question, history_text)

    llm = get_llm()
    answer = llm.invoke(prompt)

    latency_ms = (time.perf_counter() - t_start) * 1000

    return SearchResult(
        answer=answer,
        citations=_build_citation_list(docs),
        latency_ms=round(latency_ms, 2),
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
    "ask_question",
    "run_rag",
    "compare_search_modes",
]