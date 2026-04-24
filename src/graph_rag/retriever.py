"""
src/graph_rag/retriever.py — Truy vấn GraphRAGIndex theo câu hỏi.

Flow:
  1. extract_entities_from_question(): dùng LLM trích entity từ câu hỏi
  2. graph_retrieve(): lookup entity → triples → facts + citations
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document

from src.graph_rag.types import GraphRAGIndex, Triple

_logger = logging.getLogger("uvicorn.error")

# ─── Prompt extract entities ─────────────────────────────────────────────────

_EXTRACT_ENTITIES_PROMPT = """\
Bạn là hệ thống nhận diện thực thể. Từ câu hỏi dưới đây, hãy trích xuất danh sách \
thực thể chính (tên người, tên hệ thống, công nghệ, khái niệm trọng tâm) \
mà người dùng đang hỏi đến.

QUY TẮC NGHIÊM NGẶT:
1. Chỉ trả về JSON, KHÔNG giải thích, KHÔNG thêm text ngoài JSON.
2. Mỗi thực thể là một chuỗi ngắn (tên riêng, thuật ngữ).
3. Nếu không tìm thấy thực thể nào, trả về: {{"entities": []}}

Câu hỏi:
\"{question}\"

Trả về JSON theo đúng format sau (chỉ JSON, không gì khác):
{{"entities":["<entity1>","<entity2>"]}}"""


def _parse_entities_json(raw: str) -> list[str]:
    """Parse JSON list entities từ LLM output. Robust fallback."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        ents = data.get("entities", [])
        return [str(e).strip() for e in ents if str(e).strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*"entities".*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            ents = data.get("entities", [])
            return [str(e).strip() for e in ents if str(e).strip()]
        except json.JSONDecodeError:
            pass

    return []


def extract_entities_from_question(llm: Any, question: str) -> list[str]:
    """
    Dùng LLM trích xuất danh sách entity từ câu hỏi.

    Args:
        llm: LLM instance.
        question: Câu hỏi của người dùng.

    Returns:
        List tên entity (thường 1–5 items). Trả về [] nếu fail.
    """
    prompt = _EXTRACT_ENTITIES_PROMPT.format(question=question)
    try:
        raw = llm.invoke(prompt)
        raw_str = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        return _parse_entities_json(raw_str)
    except Exception as exc:
        _logger.warning("[GraphRAG] extract_entities error: %s", exc)
        return []


def _triple_to_fact_text(triple: Triple) -> str:
    """Chuyển triple thành câu "fact" dễ đọc cho LLM context."""
    return f"{triple.s} {triple.r} {triple.o}."


def graph_retrieve(
    graph_index: GraphRAGIndex,
    question: str,
    entities: list[str],
    top_k: int = 10,
) -> tuple[list[Document], list[dict]]:
    """
    Truy vấn GraphRAGIndex bằng danh sách entity đã trích xuất từ câu hỏi.

    Lookup strategy:
      - Tìm entity trong entity_to_triples (case-insensitive partial match).
      - Gộp tất cả triple liên quan, loại trùng.
      - Lấy evidence (chunk gốc) của từng triple → build Citations.
      - Context = danh sách "fact" dạng "S R O." + nội dung chunk gốc.

    Args:
        graph_index: GraphRAGIndex đã được index.
        question: Câu hỏi gốc (dùng để fallback keyword search).
        entities: List entity trích xuất từ câu hỏi.
        top_k: Số lượng triple tối đa trả về.

    Returns:
        Tuple (docs, citation_dicts):
          - docs: list Document, mỗi doc là 1 triple + evidence chunk gốc
          - citation_dicts: list dict có key content/metadata để build CitationSource
    """
    # ── 1. Lookup triples theo entity ────────────────────────────────────────
    found_triples: list[Triple] = []
    seen: set[Triple] = set()

    # Thử tìm theo entity extract được
    for entity in entities:
        entity_lower = entity.lower()
        for key, triples in graph_index.entity_to_triples.items():
            if entity_lower in key or key in entity_lower:
                for t in triples:
                    if t not in seen:
                        found_triples.append(t)
                        seen.add(t)

    # Fallback 1: nếu không match entity nào → keyword search trên triple text
    if not found_triples:
        question_lower = question.lower()
        for triple in graph_index.triples:
            fact_text = _triple_to_fact_text(triple).lower()
            if any(word in fact_text for word in question_lower.split() if len(word) > 2):
                if triple not in seen:
                    found_triples.append(triple)
                    seen.add(triple)

    # Fallback 2: câu hỏi tổng quát (tóm tắt, liệt kê, trình bày...) —
    # không match entity lẫn keyword → trả toàn bộ triples làm broad context
    # giúp LLM tổng hợp được nội dung dù không có entity cụ thể.
    if not found_triples and graph_index.triples:
        _logger.info("[GraphRAG] broad-context fallback: returning all %d triples", len(graph_index.triples))
        found_triples = list(graph_index.triples)

    # Giới hạn số triple
    found_triples = found_triples[:top_k]

    if not found_triples:
        return [], []

    # ── 2. Build Documents + citations từ evidence ────────────────────────────
    docs: list[Document] = []
    citation_dicts: list[dict] = []

    for triple in found_triples:
        fact_text = _triple_to_fact_text(triple)
        evidence_list = graph_index.evidence_by_triple.get(triple, [])

        # Lấy evidence đầu tiên (hoặc tất cả) làm context + citation
        for ev in evidence_list[:2]:   # tối đa 2 evidence chunk mỗi triple
            content = ev["content"]
            metadata = ev["metadata"]

            # Document context: fact text + chunk gốc
            full_text = f"[FACT] {fact_text}\n[EVIDENCE] {content}"
            docs.append(Document(page_content=full_text, metadata=metadata))

            # Citation: giữ nguyên chunk gốc (không thêm "[FACT]")
            citation_dicts.append({"content": content, "metadata": metadata, "score": None})

    return docs, citation_dicts
