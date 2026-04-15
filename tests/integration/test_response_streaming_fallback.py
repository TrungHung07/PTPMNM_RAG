from chatbox.domain.models import RetrievalContext
from chatbox.orchestration.llm_response import LlmResponseOrchestrator
from chatbox.orchestration.providers.ollama_adapter import OllamaAdapter
from chatbox.orchestration.providers.vllm_adapter import VllmAdapter


class _FailingProvider:
    def generate(self, prompt: str, max_tokens: int = 512) -> dict[str, str]:
        raise RuntimeError("provider unavailable")



def test_streaming_fallback_uses_secondary_provider() -> None:
    context = RetrievalContext(
        query_id="q-fallback",
        query_text="Explain retrieval",
        rag_hits=[{"chunk_id": "c-1", "score": 0.7, "text": "retrieval context"}],
        merged_evidence=[{"chunk_id": "c-1", "score": 0.7, "text": "retrieval context"}],
    )
    orchestrator = LlmResponseOrchestrator(
        primary_provider=_FailingProvider(),
        secondary_provider=VllmAdapter(model_name="fallback"),
    )

    response = orchestrator.generate(query_id="q-fallback", query_text="Explain retrieval", context=context)
    chunks = list(orchestrator.stream_text(response.answer_text, chunk_size=8))

    assert response.provider_metadata["provider"] == "vllm"
    assert "secondary_provider" in response.provider_metadata["mode"]
    assert "".join(chunks) == response.answer_text


def test_streaming_with_primary_provider() -> None:
    context = RetrievalContext(query_id="q-primary", query_text="Explain")
    orchestrator = LlmResponseOrchestrator(primary_provider=OllamaAdapter(model_name="primary"))

    response = orchestrator.generate(query_id="q-primary", query_text="Explain", context=context)
    assert response.provider_metadata["provider"] == "ollama"
