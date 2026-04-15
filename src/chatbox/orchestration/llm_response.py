from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from chatbox.domain.contracts import GenerationResponse
from chatbox.domain.models import RetrievalContext


class LlmResponseOrchestrator:
    def __init__(self, primary_provider: object, secondary_provider: object | None = None) -> None:
        self._primary_provider = primary_provider
        self._secondary_provider = secondary_provider

    def generate(self, query_id: str, query_text: str, context: RetrievalContext) -> GenerationResponse:
        prompt = self._build_prompt(query_text=query_text, context=context)
        start = perf_counter()

        provider_metadata: dict[str, str] = {}
        mode = "primary_provider"
        try:
            output = self._primary_provider.generate(prompt, max_tokens=512)
        except Exception:
            if self._secondary_provider is None:
                raise
            output = self._secondary_provider.generate(prompt, max_tokens=512)
            mode = "secondary_provider"

        latency_ms = int((perf_counter() - start) * 1000)
        provider_metadata["provider"] = output.get("provider", "unknown")
        provider_metadata["model"] = output.get("model", "unknown")
        provider_metadata["mode"] = mode

        citations = self._collect_citations(context)
        uncertainty_flags = [] if citations else ["low_evidence"]

        return GenerationResponse(
            response_id=f"resp-{uuid4()}",
            query_id=query_id,
            answer_text=output.get("text", ""),
            citations=citations,
            uncertainty_flags=uncertainty_flags,
            provider_metadata=provider_metadata,
            latency_ms=latency_ms,
        )

    def stream_text(self, text: str, chunk_size: int = 24):
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]

    def _build_prompt(self, query_text: str, context: RetrievalContext) -> str:
        evidence_lines: list[str] = []
        for item in context.merged_evidence[:5]:
            if "text" in item:
                evidence_lines.append(f"- {item['text']}")
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "- no evidence available"
        return f"Question: {query_text}\nEvidence:\n{evidence_text}\nAnswer with citations when possible."

    def _collect_citations(self, context: RetrievalContext) -> list[dict[str, str]]:
        citations: list[dict[str, str]] = []
        for item in context.merged_evidence:
            if "chunk_id" in item:
                citations.append({"chunk_id": str(item["chunk_id"]), "document_id": str(item.get("document_id", "unknown"))})
            elif "path_id" in item:
                citations.append({"chunk_id": str(item["path_id"]), "document_id": "graph"})
        return citations
