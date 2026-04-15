import asyncio
from pathlib import Path

from chatbox.graphrag.builder import GraphBuilder, InMemoryGraphStore
from chatbox.graphrag.retriever import GraphRetriever
from chatbox.ingestion.coordinator import IngestionCoordinator
from chatbox.ingestion.parsers.docx_parser import ParsedText
from chatbox.orchestration.llm_response import LlmResponseOrchestrator
from chatbox.orchestration.parallel_retrieval import ParallelRetriever
from chatbox.orchestration.providers.ollama_adapter import OllamaAdapter
from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore, RagIndexer
from chatbox.rag.retriever import RagRetriever
from chatbox.storage.sqlite_metadata import SqliteMetadataStore


class _StubParser:
    def parse(self, file_path: Path) -> ParsedText:
        return ParsedText(
            file_type="pdf",
            text=f"Alice maintains ChatBox docs from {file_path.name}",
            metadata={"source": file_path.name},
        )



def test_full_pipeline_e2e(tmp_path: Path) -> None:
    metadata_store = SqliteMetadataStore(tmp_path / "chatbox.db")
    coordinator = IngestionCoordinator(metadata_store=metadata_store, parsers={"pdf": _StubParser()})
    ingest = coordinator.ingest(file_path=tmp_path / "sample.pdf", file_type="pdf")

    document = metadata_store.get_document(ingest.document_id)
    assert document is not None

    embedding_service = EmbeddingService(dimension=64)
    vector_store = InMemoryVectorStore()
    rag_indexer = RagIndexer(embedding_service, vector_store)
    rag_retriever = RagRetriever(embedding_service, vector_store)
    rag_indexer.index_chunks([(f"{ingest.document_id}-c0", document.raw_text)], model_id="e2e-model")

    graph_store = InMemoryGraphStore()
    graph_builder = GraphBuilder(graph_store)
    graph_builder.build_from_chunks([(f"{ingest.document_id}-c0", document.raw_text)])
    graph_retriever = GraphRetriever(graph_store)

    context = asyncio.run(
        ParallelRetriever(rag_retriever=rag_retriever, graph_retriever=graph_retriever).retrieve(
            query_id="q-e2e", query_text="Alice ChatBox", top_k=5
        )
    )

    orchestrator = LlmResponseOrchestrator(primary_provider=OllamaAdapter(model_name="e2e"))
    response = orchestrator.generate(query_id="q-e2e", query_text="Alice ChatBox", context=context)

    assert response.answer_text
    assert response.citations
