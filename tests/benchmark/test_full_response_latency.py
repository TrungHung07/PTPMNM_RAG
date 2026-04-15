from time import perf_counter

from chatbox.domain.models import RetrievalContext
from chatbox.orchestration.llm_response import LlmResponseOrchestrator
from chatbox.orchestration.providers.ollama_adapter import OllamaAdapter


def test_full_response_latency_budget() -> None:
    context = RetrievalContext(
        query_id="q-full",
        query_text="Summarize local-first retrieval",
        merged_evidence=[{"chunk_id": "c1", "text": "Local-first retrieval keeps private data on-device."}],
    )
    orchestrator = LlmResponseOrchestrator(primary_provider=OllamaAdapter(model_name="latency-model"))

    start = perf_counter()
    response = orchestrator.generate(query_id="q-full", query_text=context.query_text, context=context)
    elapsed = perf_counter() - start

    assert response.answer_text
    assert elapsed < 0.2
