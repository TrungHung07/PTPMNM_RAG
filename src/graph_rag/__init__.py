"""
src/graph_rag — Graph RAG module.

Cung cấp pipeline RAG dựa trên knowledge graph (triples):
  - extractor: trích xuất (Subject, Relation, Object) từ chunk
  - indexer:   xây dựng GraphRAGIndex (graph + evidence store)
  - retriever: truy vấn graph theo câu hỏi, trả Document "facts"
  - pipeline:  orchestrate full flow → SearchResult (cùng shape với Standard RAG)

Standard RAG (src/rag/) không bị ảnh hưởng.
"""
from src.graph_rag.types import GraphRAGIndex, Triple
from src.graph_rag.indexer import build_graph_index, merge_graph_indices
from src.graph_rag.pipeline import run_graph_rag

__all__ = [
    "GraphRAGIndex",
    "Triple",
    "build_graph_index",
    "merge_graph_indices",
    "run_graph_rag",
]
