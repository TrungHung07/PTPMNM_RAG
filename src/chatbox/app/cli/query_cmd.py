from __future__ import annotations

import asyncio

import typer

from chatbox.graphrag.builder import GraphBuilder, InMemoryGraphStore
from chatbox.graphrag.retriever import GraphRetriever
from chatbox.orchestration.llm_response import LlmResponseOrchestrator
from chatbox.orchestration.parallel_retrieval import ParallelRetriever
from chatbox.orchestration.providers.ollama_adapter import OllamaAdapter
from chatbox.orchestration.providers.vllm_adapter import VllmAdapter
from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever

app = typer.Typer(help="Run local query in rag or hybrid mode")


@app.command("run")
def run_query(query: str, mode: str = "hybrid", stream: bool = True) -> None:
    embedding_service = EmbeddingService(dimension=64)
    vector_store = InMemoryVectorStore()
    rag_indexer = RagIndexer(embedding_service, vector_store)
    rag_retriever = RagRetriever(embedding_service, vector_store)

    graph_retriever = None
    if mode == "hybrid":
        graph_store = InMemoryGraphStore()
        graph_builder = GraphBuilder(graph_store)
        graph_builder.build_from_chunks([("seed-1", "ChatBox supports local-first retrieval")])
        graph_retriever = GraphRetriever(graph_store)

    rag_indexer.index_chunks([("seed-1", "ChatBox supports local-first retrieval")], model_id="cli-model")
    context = asyncio.run(
        ParallelRetriever(rag_retriever=rag_retriever, graph_retriever=graph_retriever).retrieve(
            query_id="cli-q",
            query_text=query,
            top_k=5,
        )
    )

    orchestrator = LlmResponseOrchestrator(
        primary_provider=OllamaAdapter(model_name="local-primary"),
        secondary_provider=VllmAdapter(model_name="local-fallback"),
    )
    response = orchestrator.generate(query_id="cli-q", query_text=query, context=context)

    if stream:
        for piece in orchestrator.stream_text(response.answer_text, chunk_size=18):
            typer.echo(piece, nl=False)
        typer.echo("")
        return

    typer.echo(response.answer_text)
