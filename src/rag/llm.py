import os
from langchain_community.llms import Ollama
from functools import lru_cache

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@lru_cache(maxsize=1)
def get_llm():
    return Ollama(model="qwen2.5:3b", base_url=OLLAMA_BASE_URL)
