from chatbox.domain.models import RetrievalContext
from chatbox.orchestration.merger import merge_rag_context
from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever


def test_rag_query_to_context_assembly() -> None:
    embedding_service = EmbeddingService(dimension=8)
    store = InMemoryVectorStore()
    indexer = RagIndexer(embedding_service=embedding_service, vector_store=store)
    retriever = RagRetriever(embedding_service=embedding_service, vector_store=store)

    indexer.index_chunks(
        [
            ("c1", "local first rag architecture"),
            ("c2", "graph retrieval and entities"),
        ],
        model_id="integration-model",
    )

    rag_hits = retriever.retrieve("rag architecture", top_k=2)
    context = merge_rag_context(query_id="q-rag", query_text="rag architecture", rag_hits=rag_hits)

    assert isinstance(context, RetrievalContext)
    assert context.query_id == "q-rag"
    assert context.rag_hits
