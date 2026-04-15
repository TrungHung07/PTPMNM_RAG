import asyncio

from chatbox.graphrag.builder import GraphBuilder, InMemoryGraphStore
from chatbox.graphrag.retriever import GraphRetriever
from chatbox.orchestration.parallel_retrieval import ParallelRetriever
from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever


def test_parallel_retrieval_and_fusion() -> None:
    embedding_service = EmbeddingService(dimension=32)
    vector_store = InMemoryVectorStore()
    rag_indexer = RagIndexer(embedding_service, vector_store)
    rag_retriever = RagRetriever(embedding_service, vector_store)

    graph_store = InMemoryGraphStore()
    graph_builder = GraphBuilder(graph_store)
    graph_retriever = GraphRetriever(graph_store)

    rag_indexer.index_chunks([("c1", "Alice builds local rag systems")], model_id="hybrid")
    graph_builder.build_from_chunks([("c1", "Alice builds local rag systems")])

    orchestrator = ParallelRetriever(rag_retriever=rag_retriever, graph_retriever=graph_retriever)
    context = asyncio.run(orchestrator.retrieve(query_id="q1", query_text="alice rag", top_k=3))

    assert context.merged_evidence
    assert context.rag_hits
    assert context.graph_hits
