"""
src/graph_rag/indexer.py — Xây dựng GraphRAGIndex từ danh sách Document chunks.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from src.graph_rag.types import GraphRAGIndex, Triple
from src.graph_rag.extractor import extract_triples_from_chunk

_logger = logging.getLogger("uvicorn.error")


def build_graph_index(
    chunks: list[Document],
    llm: Any,
    max_triples_per_chunk: int = 10,
) -> GraphRAGIndex:
    """
    Xây dựng GraphRAGIndex từ danh sách Document chunk.

    Với mỗi chunk:
    1. Gọi extractor để lấy list Triple.
    2. Với mỗi triple, lưu evidence (content + metadata của chunk gốc).
    3. Index entity → triple để lookup nhanh khi query.

    Args:
        chunks: Danh sách Document chunk đã được split (từ RAGIndex.chunks).
        llm: LLM instance để trích xuất triple.
        max_triples_per_chunk: Giới hạn số triple trích xuất mỗi chunk.

    Returns:
        GraphRAGIndex sẵn sàng cho retrieval.
    """
    index = GraphRAGIndex()
    total_triples = 0

    for i, chunk in enumerate(chunks):
        try:
            triples = extract_triples_from_chunk(llm, chunk, max_triples=max_triples_per_chunk)
        except Exception as exc:
            _logger.warning("[GraphRAG] indexer: chunk %d extraction error: %s", i, exc)
            triples = []

        for triple in triples:
            index.add_triple(
                triple=triple,
                content=chunk.page_content,
                metadata=chunk.metadata or {},
            )
            total_triples += 1

    _logger.info(
        "[GraphRAG] build_graph_index: %d chunks → %d triples, %d unique entities",
        len(chunks),
        total_triples,
        len(index.entity_to_triples),
    )
    return index


def merge_graph_indices(indices: list[GraphRAGIndex]) -> GraphRAGIndex:
    """
    Gộp nhiều GraphRAGIndex (mỗi file) thành một index để query chung.

    Union tất cả triples + evidence. Triple trùng (cùng s/r/o) thì gộp evidence list.

    Args:
        indices: Danh sách GraphRAGIndex cần gộp.

    Returns:
        GraphRAGIndex đã gộp.
    """
    if not indices:
        raise ValueError("merge_graph_indices: indices rỗng")
    if len(indices) == 1:
        return indices[0]

    merged = GraphRAGIndex()
    for idx in indices:
        for triple in idx.triples:
            evidence_list = idx.evidence_by_triple.get(triple, [])
            for ev in evidence_list:
                merged.add_triple(triple, ev["content"], ev["metadata"])

    return merged
