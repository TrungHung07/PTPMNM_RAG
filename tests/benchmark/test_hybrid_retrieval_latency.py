import asyncio
from time import perf_counter

from chatbox.graphrag.builder import GraphBuilder, InMemoryGraphStore
from chatbox.graphrag.retriever import GraphRetriever
from chatbox.orchestration.parallel_retrieval import ParallelRetriever
from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever


def test_hybrid_retrieval_latency_budget() -> None:
    embedding_service = EmbeddingService(dimension=64)
    vector_store = InMemoryVectorStore()
    rag_indexer = RagIndexer(embedding_service, vector_store)
    rag_retriever = RagRetriever(embedding_service, vector_store)

    graph_store = InMemoryGraphStore()
    graph_builder = GraphBuilder(graph_store)
    graph_retriever = GraphRetriever(graph_store)

    chunks = [(f"c-{i}", f"Alice manages team {i} at Acme") for i in range(100)]
    rag_indexer.index_chunks(chunks, model_id="hybrid-bench")
    graph_builder.build_from_chunks(chunks)

    orchestrator = ParallelRetriever(rag_retriever=rag_retriever, graph_retriever=graph_retriever)

    start = perf_counter()
    context = asyncio.run(orchestrator.retrieve(query_id="q-bench", query_text="Alice Acme", top_k=5))
    elapsed = perf_counter() - start

    assert context.merged_evidence
    assert elapsed < 0.5
