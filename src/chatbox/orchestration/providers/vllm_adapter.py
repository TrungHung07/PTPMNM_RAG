from __future__ import annotations


class VllmAdapter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512) -> dict[str, str]:
        preview = prompt[: max_tokens // 2].strip()
        return {
            "text": f"[vllm:{self.model_name}] {preview}" if preview else "[vllm] no context",
            "provider": "vllm",
            "model": self.model_name,
        }
