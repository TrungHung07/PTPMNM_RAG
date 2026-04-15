from __future__ import annotations

from chatbox.domain.models import RetrievalContext
from chatbox.storage.ports import GraphHit, RagHit


def merge_rag_context(query_id: str, query_text: str, rag_hits: list[RagHit]) -> RetrievalContext:
    rag_payload = [
        {"chunk_id": hit.chunk_id, "score": hit.score, "text": hit.text}
        for hit in rag_hits
    ]
    return RetrievalContext(
        query_id=query_id,
        query_text=query_text,
        rag_hits=rag_payload,
        merged_evidence=rag_payload,
        degraded_mode=None,
    )


def merge_hybrid_context(
    query_id: str,
    query_text: str,
    rag_hits: list[RagHit],
    graph_hits: list[GraphHit],
    degraded_mode: str | None = None,
) -> RetrievalContext:
    rag_payload = [
        {"chunk_id": hit.chunk_id, "score": hit.score, "text": hit.text}
        for hit in rag_hits
    ]
    graph_payload = [
        {"path_id": hit.path_id, "score": hit.score, "evidence_chunk_ids": hit.evidence_chunk_ids}
        for hit in graph_hits
    ]
    merged_evidence = sorted(
        rag_payload + graph_payload,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )
    return RetrievalContext(
        query_id=query_id,
        query_text=query_text,
        rag_hits=rag_payload,
        graph_hits=graph_payload,
        merged_evidence=merged_evidence,
        degraded_mode=degraded_mode,
    )
