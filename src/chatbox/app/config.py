from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    fallback_mode: str = "hybrid"
    retrieval_timeout_ms: int = 2500
    generation_timeout_ms: int = 8000


def load_config() -> AppConfig:
    return AppConfig(
        fallback_mode=os.getenv("CHATBOX_FALLBACK_MODE", "hybrid"),
        retrieval_timeout_ms=int(os.getenv("CHATBOX_RETRIEVAL_TIMEOUT_MS", "2500")),
        generation_timeout_ms=int(os.getenv("CHATBOX_GENERATION_TIMEOUT_MS", "8000")),
    )
