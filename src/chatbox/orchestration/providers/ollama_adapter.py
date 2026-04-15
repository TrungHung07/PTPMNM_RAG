from __future__ import annotations


class OllamaAdapter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512) -> dict[str, str]:
        preview = prompt[: max_tokens // 2].strip()
        return {
            "text": f"[ollama:{self.model_name}] {preview}" if preview else "[ollama] no context",
            "provider": "ollama",
            "model": self.model_name,
        }
