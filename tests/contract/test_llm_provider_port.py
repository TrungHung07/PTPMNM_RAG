from chatbox.orchestration.providers.ollama_adapter import OllamaAdapter
from chatbox.orchestration.providers.vllm_adapter import VllmAdapter


def test_llm_provider_contract_shape() -> None:
    ollama = OllamaAdapter(model_name="test-model")
    vllm = VllmAdapter(model_name="test-model")

    ollama_response = ollama.generate("hello", max_tokens=64)
    vllm_response = vllm.generate("hello", max_tokens=64)

    assert "text" in ollama_response
    assert "provider" in ollama_response
    assert "text" in vllm_response
    assert "provider" in vllm_response
