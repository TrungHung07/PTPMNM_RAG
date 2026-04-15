from __future__ import annotations

import asyncio

from chatbox.domain.models import RetrievalContext
from chatbox.graphrag.retriever import GraphRetriever
from chatbox.orchestration.merger import merge_hybrid_context, merge_rag_context
from chatbox.rag.retriever import RagRetriever


class ParallelRetriever:
    def __init__(self, rag_retriever: RagRetriever, graph_retriever: GraphRetriever | None = None) -> None:
        self._rag_retriever = rag_retriever
        self._graph_retriever = graph_retriever

    async def retrieve(self, query_id: str, query_text: str, top_k: int = 5) -> RetrievalContext:
        if self._graph_retriever is None:
            rag_hits = self._rag_retriever.retrieve(query_text, top_k=top_k)
            return merge_rag_context(query_id=query_id, query_text=query_text, rag_hits=rag_hits)

        rag_task = asyncio.to_thread(self._rag_retriever.retrieve, query_text, top_k)
        graph_task = asyncio.to_thread(self._graph_retriever.retrieve, query_text, top_k)

        rag_result, graph_result = await asyncio.gather(rag_task, graph_task, return_exceptions=True)

        rag_hits = [] if isinstance(rag_result, Exception) else rag_result
        graph_hits = [] if isinstance(graph_result, Exception) else graph_result

        degraded_mode = None
        if isinstance(rag_result, Exception) and isinstance(graph_result, Exception):
            degraded_mode = "retrieval_unavailable"
        elif isinstance(rag_result, Exception):
            degraded_mode = "graph_only"
        elif isinstance(graph_result, Exception):
            degraded_mode = "rag_only"

        return merge_hybrid_context(
            query_id=query_id,
            query_text=query_text,
            rag_hits=rag_hits,
            graph_hits=graph_hits,
            degraded_mode=degraded_mode,
        )
