from time import perf_counter

from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever


def test_rag_retrieval_latency_budget() -> None:
    embedding_service = EmbeddingService(dimension=64)
    store = InMemoryVectorStore()
    indexer = RagIndexer(embedding_service=embedding_service, vector_store=store)
    retriever = RagRetriever(embedding_service=embedding_service, vector_store=store)

    chunks = [(f"c-{i}", f"knowledge chunk number {i}") for i in range(200)]
    indexer.index_chunks(chunks=chunks, model_id="bench-rag")

    start = perf_counter()
    hits = retriever.retrieve("knowledge number 42", top_k=5)
    elapsed = perf_counter() - start

    assert len(hits) == 5
    assert elapsed < 0.2
