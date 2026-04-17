from langchain_community.llms import Ollama


def get_llm():
    return Ollama(model="qwen2.5:3b")