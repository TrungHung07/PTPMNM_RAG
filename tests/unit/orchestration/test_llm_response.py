from chatbox.domain.models import RetrievalContext
from chatbox.orchestration.llm_response import LlmResponseOrchestrator
from chatbox.orchestration.providers.ollama_adapter import OllamaAdapter


def test_llm_response_includes_citations_and_uncertainty() -> None:
    context = RetrievalContext(
        query_id="q1",
        query_text="What is local RAG?",
        rag_hits=[{"chunk_id": "c1", "score": 0.9, "text": "Local RAG keeps data on device."}],
        graph_hits=[],
        merged_evidence=[{"chunk_id": "c1", "score": 0.9, "text": "Local RAG keeps data on device."}],
    )
    orchestrator = LlmResponseOrchestrator(primary_provider=OllamaAdapter(model_name="test"))

    response = orchestrator.generate(query_id="q1", query_text="What is local RAG?", context=context)

    assert response.citations
    assert response.citations[0]["chunk_id"] == "c1"
    assert isinstance(response.uncertainty_flags, list)
