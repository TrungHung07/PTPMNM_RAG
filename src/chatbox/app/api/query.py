from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from chatbox.domain.contracts import QueryAnswerResponse, QueryContextResponse, QueryRequest
from chatbox.graphrag.retriever import GraphRetriever
from chatbox.orchestration.llm_response import LlmResponseOrchestrator
from chatbox.orchestration.merger import merge_rag_context
from chatbox.orchestration.parallel_retrieval import ParallelRetriever
from chatbox.rag.retriever import RagRetriever


def create_query_router(
    rag_retriever: RagRetriever,
    graph_retriever: GraphRetriever | None = None,
    llm_orchestrator: LlmResponseOrchestrator | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["query"])
    parallel_retriever = ParallelRetriever(rag_retriever=rag_retriever, graph_retriever=graph_retriever)

    def _resolve_context(query_id: str, request: QueryRequest):
        if request.mode == "rag":
            rag_hits = rag_retriever.retrieve(request.query_text, top_k=request.top_k)
            return merge_rag_context(query_id=query_id, query_text=request.query_text, rag_hits=rag_hits)
        return asyncio.run(
            parallel_retriever.retrieve(query_id=query_id, query_text=request.query_text, top_k=request.top_k)
        )

    @router.post("/query/context", response_model=QueryContextResponse)
    def query_context(request: QueryRequest) -> QueryContextResponse:
        query_id = f"q-{uuid4()}"
        context = _resolve_context(query_id=query_id, request=request)
        return QueryContextResponse(query_id=query_id, retrieval_context=context)

    @router.post("/query", response_model=QueryAnswerResponse)
    def query(request: QueryRequest):
        query_id = f"q-{uuid4()}"
        context = _resolve_context(query_id=query_id, request=request)

        if llm_orchestrator is None:
            text = "No LLM provider configured for generation"
            if request.stream:
                return StreamingResponse(iter([text]), media_type="text/plain")
            return QueryAnswerResponse(
                query_id=query_id,
                answer_text=text,
                retrieval_context=context,
            )

        generation = llm_orchestrator.generate(
            query_id=query_id,
            query_text=request.query_text,
            context=context,
        )

        if request.stream:
            return StreamingResponse(
                llm_orchestrator.stream_text(generation.answer_text),
                media_type="text/plain",
            )

        return QueryAnswerResponse(
            query_id=query_id,
            answer_text=generation.answer_text,
            citations=generation.citations,
            uncertainty_flags=generation.uncertainty_flags,
            provider_metadata=generation.provider_metadata,
            retrieval_context=context,
        )

    return router
