from chatbox.rag.indexer import InMemoryVectorStore
from chatbox.storage.ports import RagHit


def test_vector_store_port_upsert_and_search() -> None:
    store = InMemoryVectorStore()
    store.upsert_embeddings(
        [
            {
                "embedding_id": "e1",
                "chunk_id": "c1",
                "model_id": "test-model",
                "vector": [1.0, 0.0],
                "dimension": 2,
            },
            {
                "embedding_id": "e2",
                "chunk_id": "c2",
                "model_id": "test-model",
                "vector": [0.0, 1.0],
                "dimension": 2,
            },
        ]
    )

    results = store.search(query_vector=[0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RagHit)
    assert results[0].chunk_id == "c1"
