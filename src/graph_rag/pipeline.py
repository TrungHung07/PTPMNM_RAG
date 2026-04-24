"""
src/graph_rag/pipeline.py — Orchestrate Graph RAG full flow.

run_graph_rag() trả SearchResult cùng shape với Standard RAG để
/ask endpoint có thể dùng chung code path append_message + response.
"""
from __future__ import annotations

import logging
from typing import Any

from src.graph_rag.types import GraphRAGIndex
from src.graph_rag.retriever import extract_entities_from_question, graph_retrieve
from src.models import CitationSource, SearchResult

_logger = logging.getLogger("uvicorn.error")

_HISTORY_WINDOW = 5


def _format_chat_history(chat_history: list) -> str:
    """Format lịch sử hội thoại thành chuỗi Human/AI."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        lines.append(f"Human: {msg['question']}")
        lines.append(f"AI: {msg['answer']}")
    return "\n".join(lines)


_GRAPH_RAG_PROMPT = """\
BẠN LÀ MỘT TRỢ LÝ AI CHUYÊN NGHIỆP. BẠN TUYỆT ĐỐI CHỈ ĐƯỢC DÙNG TIẾNG VIỆT HOẶC TIẾNG ANH.

QUY TẮC BẮT BUỘC:
1. KHÔNG sử dụng bất kỳ từ ngữ nào từ tiếng Malay, Indonesian.
2. Trả lời dựa trên các "Sự kiện tri thức" và "Bằng chứng" bên dưới.
3. Nếu không có thông tin phù hợp, trả lời "{no_info_phrase}".
4. Trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng.

Sự kiện tri thức và bằng chứng từ tài liệu:
{context}

{history_section}

Câu hỏi hiện tại: {question}
TRẢ LỜI (CHỈ DÙNG TIẾNG VIỆT HOẶC TIẾNG ANH):"""


def run_graph_rag(
    graph_index: GraphRAGIndex,
    question: str,
    chat_history: list,
    llm: Any,
    *,
    no_info_phrase: str = "Không tìm thấy thông tin trong tài liệu.",
    top_k: int = 20,
) -> SearchResult:
    """
    Thực thi Graph RAG pipeline đầy đủ.

    Flow:
      1. Trích entity từ câu hỏi bằng LLM.
      2. Lookup graph → facts + evidence citations.
      3. Nếu không có facts → trả no_info_phrase, citations=[].
      4. Build prompt (facts + history) → gọi LLM → answer.
      5. Đóng gói thành SearchResult (cùng shape Standard RAG).

    Args:
        graph_index: GraphRAGIndex đã index.
        question: Câu hỏi của người dùng.
        chat_history: Lịch sử hội thoại [{question, answer}].
        llm: LLM instance.
        no_info_phrase: Chuỗi trả lời khi không tìm thấy thông tin.
        top_k: Số triple tối đa đưa vào context (mặc định 20).

    Returns:
        SearchResult(answer, citations) — cùng shape với Standard RAG.
    """
    # Detect câu hỏi tổng quát / summarization → lấy nhiều triple hơn
    _SUMMARY_KEYWORDS = {
        "tóm tắt", "tóm lược", "liệt kê", "trình bày", "mô tả", "giới thiệu",
        "nêu", "tổng hợp", "tổng quan", "overview", "summary", "summarize",
        "summarise", "list", "describe", "outline", "explain all",
    }
    question_lower = question.lower()
    is_summary_query = any(kw in question_lower for kw in _SUMMARY_KEYWORDS)
    effective_top_k = 9999 if is_summary_query else top_k

    # ── 1. Extract entities ──────────────────────────────────────────────────
    try:
        entities = extract_entities_from_question(llm, question)
    except Exception as exc:
        _logger.warning("[GraphRAG] entity extraction failed: %s", exc)
        entities = []

    _logger.info(
        "[GraphRAG] question=%r entities=%s is_summary=%s top_k=%s",
        question, entities, is_summary_query, effective_top_k,
    )

    # ── 2. Retrieve from graph ───────────────────────────────────────────────
    try:
        docs, citation_dicts = graph_retrieve(
            graph_index=graph_index,
            question=question,
            entities=entities,
            top_k=effective_top_k,
        )
    except Exception as exc:
        _logger.warning("[GraphRAG] graph_retrieve failed: %s", exc)
        docs, citation_dicts = [], []

    # ── 3. No context → trả no_info ─────────────────────────────────────────
    if not docs:
        _logger.info("[GraphRAG] no facts found for question=%r", question)
        return SearchResult(answer=no_info_phrase, citations=[])

    # ── 4. Build prompt & call LLM ───────────────────────────────────────────
    context = "\n\n".join(doc.page_content for doc in docs)
    recent_history = chat_history[-_HISTORY_WINDOW:]
    history_text = _format_chat_history(recent_history)
    history_section = (
        f"Lịch sử hội thoại trước đó (để hiểu ngữ cảnh follow-up):\n{history_text}"
        if history_text else ""
    )

    prompt = _GRAPH_RAG_PROMPT.format(
        context=context,
        question=question,
        no_info_phrase=no_info_phrase,
        history_section=history_section,
    )

    try:
        raw_answer = llm.invoke(prompt)
        answer = raw_answer if isinstance(raw_answer, str) else getattr(raw_answer, "content", str(raw_answer))
    except Exception as exc:
        _logger.error("[GraphRAG] LLM call failed: %s", exc)
        answer = no_info_phrase
        citation_dicts = []

    # ── 5. Build citations ───────────────────────────────────────────────────
    if no_info_phrase in answer:
        citations = []
    else:
        citations = [
            CitationSource(
                content=c["content"],
                metadata=c["metadata"],
                score=c.get("score"),
            )
            for c in citation_dicts
        ]

    return SearchResult(answer=answer, citations=citations)
