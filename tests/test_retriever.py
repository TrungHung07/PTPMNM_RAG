"""
Unit tests cho src/rag/retriever.py

Kiểm tra 3 factory functions:
  - build_vector_retriever(): wrap FAISS thành retriever đúng chuẩn
  - build_bm25_retriever(): tạo BM25 index từ chunks, trả về đúng k docs
  - build_hybrid_retriever(): kết hợp hai retriever, validate weights

Các test này KHÔNG gọi LLM và KHÔNG cần database.
Vector retriever test cần embedding model (HuggingFace) — dùng fixture nhỏ.
"""
import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag.retriever import build_vector_retriever, build_bm25_retriever, build_hybrid_retriever


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_chunks() -> list[Document]:
    """10 Document mẫu đại diện cho kết quả chunking từ tài liệu thực tế."""
    return [
        Document(
            page_content=f"Đây là đoạn văn số {i}. Nội dung thảo luận về chủ đề {topic}.",
            metadata={"page": i, "source": "test.pdf", "chunk_index": 0}
        )
        for i, topic in enumerate([
            "trí tuệ nhân tạo", "học máy", "mạng nơ-ron", "xử lý ngôn ngữ tự nhiên",
            "thị giác máy tính", "thuật toán BM25", "vector search", "RAG pipeline",
            "embedding model", "FAISS index"
        ], start=1)
    ]


@pytest.fixture(scope="module")
def small_vectorstore(sample_chunks):
    """FAISS vectorstore nhỏ dùng real embedding model để test."""
    from src.rag.embedding import get_embedding
    from src.rag.vectorstore import create_vectorstore
    return create_vectorstore(sample_chunks, get_embedding())


# ── BM25 Retriever ─────────────────────────────────────────────────────────────

class TestBuildBM25Retriever:
    """Kiểm tra build_bm25_retriever()."""

    def test_returns_base_retriever(self, sample_chunks):
        """Hàm phải trả về BaseRetriever (tương thích interface LangChain)."""
        retriever = build_bm25_retriever(sample_chunks, k=3)
        assert isinstance(retriever, BaseRetriever)

    def test_respects_k_limit(self, sample_chunks):
        """Retriever phải trả về tối đa k documents."""
        k = 3
        retriever = build_bm25_retriever(sample_chunks, k=k)
        results = retriever.invoke("trí tuệ nhân tạo")
        assert len(results) <= k

    def test_keyword_match(self, sample_chunks):
        """BM25 phải ưu tiên document chứa từ khoá chính xác cao hơn."""
        retriever = build_bm25_retriever(sample_chunks, k=5)
        results = retriever.invoke("BM25")
        # Document "thuật toán BM25" phải nằm trong top results
        contents = [doc.page_content for doc in results]
        assert any("BM25" in c for c in contents)

    def test_metadata_preserved_in_results(self, sample_chunks):
        """Documents trả về phải giữ nguyên metadata gốc (page, source)."""
        retriever = build_bm25_retriever(sample_chunks, k=3)
        results = retriever.invoke("vector search")
        for doc in results:
            assert "page" in doc.metadata
            assert "source" in doc.metadata

    def test_default_k_is_four(self, sample_chunks):
        """k mặc định phải là 4."""
        retriever = build_bm25_retriever(sample_chunks)
        results = retriever.invoke("học máy")
        assert len(results) <= 4


# ── Vector Retriever ───────────────────────────────────────────────────────────

class TestBuildVectorRetriever:
    """Kiểm tra build_vector_retriever()."""

    def test_returns_base_retriever(self, small_vectorstore):
        """Hàm phải trả về BaseRetriever."""
        retriever = build_vector_retriever(small_vectorstore, k=3)
        assert isinstance(retriever, BaseRetriever)

    def test_respects_k_limit(self, small_vectorstore):
        """Retriever phải trả về tối đa k documents."""
        k = 2
        retriever = build_vector_retriever(small_vectorstore, k=k)
        results = retriever.invoke("mạng nơ-ron")
        assert len(results) <= k

    def test_metadata_preserved_in_results(self, small_vectorstore):
        """Documents trả về phải giữ nguyên metadata gốc."""
        retriever = build_vector_retriever(small_vectorstore, k=3)
        results = retriever.invoke("embedding")
        for doc in results:
            assert "page" in doc.metadata
            assert "source" in doc.metadata


# ── Hybrid Retriever ───────────────────────────────────────────────────────────

class TestBuildHybridRetriever:
    """Kiểm tra build_hybrid_retriever()."""

    def test_returns_base_retriever(self, small_vectorstore, sample_chunks):
        """Hàm phải trả về BaseRetriever."""
        retriever = build_hybrid_retriever(small_vectorstore, sample_chunks, k=3)
        assert isinstance(retriever, BaseRetriever)

    def test_returns_results(self, small_vectorstore, sample_chunks):
        """Phải trả về ít nhất 1 document."""
        retriever = build_hybrid_retriever(small_vectorstore, sample_chunks, k=4)
        results = retriever.invoke("học máy")
        assert len(results) >= 1

    def test_invalid_bm25_weight_raises(self, small_vectorstore, sample_chunks):
        """bm25_weight ngoài [0.0, 1.0] phải raise ValueError."""
        with pytest.raises(ValueError, match="bm25_weight"):
            build_hybrid_retriever(small_vectorstore, sample_chunks, bm25_weight=1.5)

        with pytest.raises(ValueError, match="bm25_weight"):
            build_hybrid_retriever(small_vectorstore, sample_chunks, bm25_weight=-0.1)

    def test_boundary_weights(self, small_vectorstore, sample_chunks):
        """Weight 0.0 và 1.0 (biên) phải hợp lệ — không raise."""
        retriever_pure_vector = build_hybrid_retriever(
            small_vectorstore, sample_chunks, bm25_weight=0.0
        )
        retriever_pure_bm25 = build_hybrid_retriever(
            small_vectorstore, sample_chunks, bm25_weight=1.0
        )
        assert retriever_pure_vector is not None
        assert retriever_pure_bm25 is not None

    def test_different_weights_may_change_results(self, small_vectorstore, sample_chunks):
        """
        Hai bộ weight khác nhau có thể trả về kết quả khác nhau.
        Test này không đảm bảo luôn khác — phụ thuộc dữ liệu — nhưng không được crash.
        """
        retriever_bm25_heavy = build_hybrid_retriever(
            small_vectorstore, sample_chunks, k=3, bm25_weight=0.8
        )
        retriever_vector_heavy = build_hybrid_retriever(
            small_vectorstore, sample_chunks, k=3, bm25_weight=0.2
        )
        results_bm25 = retriever_bm25_heavy.invoke("BM25 algorithm")
        results_vector = retriever_vector_heavy.invoke("BM25 algorithm")
        # Cả hai phải trả về kết quả hợp lệ
        assert len(results_bm25) >= 1
        assert len(results_vector) >= 1
