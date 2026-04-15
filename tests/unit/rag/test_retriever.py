from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever


def test_rag_retriever_returns_ranked_hits() -> None:
    embedding_service = EmbeddingService(dimension=8)
    store = InMemoryVectorStore()
    indexer = RagIndexer(embedding_service=embedding_service, vector_store=store)
    retriever = RagRetriever(embedding_service=embedding_service, vector_store=store)

    indexer.index_chunks(
        [
            ("c1", "python data classes and typing"),
            ("c2", "graph traversal shortest paths"),
        ],
        model_id="unit-test-model",
    )

    hits = retriever.retrieve("python typing", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
