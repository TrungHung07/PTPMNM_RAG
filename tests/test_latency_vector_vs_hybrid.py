"""
Test latency: Vector Search vs Hybrid Search.

So sánh thời gian retrieval giữa hai chế độ:
  - Vector Search (FAISS only)
  - Hybrid Search (BM25 + FAISS qua RRF)

Các test đo latency trung bình, P95, và overhead của Hybrid so với Vector.
Không gọi LLM, chỉ đo retrieval layer.
"""
import time
import statistics

import pytest
from langchain_core.documents import Document

from src.rag.retriever import build_vector_retriever, build_hybrid_retriever


# ── Fixtures ───────────────────────────────────────────────────────────────────

QUERIES = [
    "Chạy bộ mang lại lợi ích gì cho tim mạch?",
    "Lợi ích tinh thần của chạy bộ là gì?",
    "Những nguyên tắc vàng cho người mới bắt đầu chạy bộ là gì?",
    "Đối thủ lớn nhất trên đường chạy là ai?",
    "Kỹ thuật chạy bộ đúng cách",
    "Dinh dưỡng cho người chạy bộ",
    "Trang phục phù hợp khi chạy bộ",
    "Lịch tập chạy bộ cho người mới",
]


@pytest.fixture(scope="module")
def sample_chunks() -> list[Document]:
    """20 Document mẫu mô phỏng kết quả chunking."""
    topics = [
        "lợi ích tim mạch", "lợi ích tinh thần", "kỹ thuật chạy",
        "dinh dưỡng", "trang phục", "lịch tập", "chấn thương",
        "khởi động", "giãn cơ", "tốc độ", "nhịp thở", "tư thế",
        "giày chạy", "đường chạy", "thời tiết", "mục tiêu",
        "động lực", "cộng đồng", "giải chạy", "phục hồi",
    ]
    return [
        Document(
            page_content=f"Đoạn văn về {topic}. " * 20,
            metadata={"page": i + 1, "source": "chay_bo.pdf", "chunk_index": 0},
        )
        for i, topic in enumerate(topics)
    ]


@pytest.fixture(scope="module")
def vectorstore(sample_chunks):
    """FAISS vectorstore từ sample chunks."""
    from src.rag.embedding import get_embedding
    from src.rag.vectorstore import create_vectorstore
    return create_vectorstore(sample_chunks, get_embedding())


@pytest.fixture(scope="module")
def vector_retriever(vectorstore):
    return build_vector_retriever(vectorstore, k=4)


@pytest.fixture(scope="module")
def hybrid_retriever(vectorstore, sample_chunks):
    return build_hybrid_retriever(vectorstore, sample_chunks, k=4, bm25_weight=0.5)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _measure_latency(retriever, queries: list[str], runs: int = 3) -> list[float]:
    """Đo latency (ms) cho mỗi query, lấy trung bình qua nhiều lần chạy."""
    latencies = []
    for q in queries:
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            retriever.invoke(q)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        latencies.append(statistics.mean(times))
    return latencies


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestVectorSearchLatency:
    """Đo latency cơ bản của Vector Search."""

    def test_vector_returns_results(self, vector_retriever):
        """Vector retriever phải trả về kết quả."""
        results = vector_retriever.invoke(QUERIES[0])
        assert len(results) >= 1

    def test_vector_latency_under_threshold(self, vector_retriever):
        """Latency trung bình của Vector Search phải dưới 200ms."""
        latencies = _measure_latency(vector_retriever, QUERIES)
        avg = statistics.mean(latencies)
        assert avg < 200, f"Vector avg latency {avg:.1f}ms vượt ngưỡng 200ms"

    def test_vector_p95_under_threshold(self, vector_retriever):
        """P95 latency của Vector Search phải dưới 300ms."""
        latencies = _measure_latency(vector_retriever, QUERIES)
        latencies_sorted = sorted(latencies)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
        assert p95 < 300, f"Vector P95 latency {p95:.1f}ms vượt ngưỡng 300ms"


class TestHybridSearchLatency:
    """Đo latency cơ bản của Hybrid Search."""

    def test_hybrid_returns_results(self, hybrid_retriever):
        """Hybrid retriever phải trả về kết quả."""
        results = hybrid_retriever.invoke(QUERIES[0])
        assert len(results) >= 1

    def test_hybrid_latency_under_threshold(self, hybrid_retriever):
        """Latency trung bình của Hybrid Search phải dưới 500ms."""
        latencies = _measure_latency(hybrid_retriever, QUERIES)
        avg = statistics.mean(latencies)
        assert avg < 500, f"Hybrid avg latency {avg:.1f}ms vượt ngưỡng 500ms"


class TestLatencyComparison:
    """So sánh latency giữa Vector và Hybrid Search."""

    def test_hybrid_slower_than_vector(self, vector_retriever, hybrid_retriever):
        """Hybrid Search phải chậm hơn Vector Search (do BM25 + RRF overhead)."""
        vec_latencies = _measure_latency(vector_retriever, QUERIES)
        hyb_latencies = _measure_latency(hybrid_retriever, QUERIES)

        vec_avg = statistics.mean(vec_latencies)
        hyb_avg = statistics.mean(hyb_latencies)

        # Hybrid phải chậm hơn (overhead > 0)
        assert hyb_avg > vec_avg, (
            f"Hybrid ({hyb_avg:.1f}ms) không chậm hơn Vector ({vec_avg:.1f}ms)"
        )

    def test_hybrid_overhead_within_limit(self, vector_retriever, hybrid_retriever):
        """Overhead của Hybrid so với Vector không được vượt quá 5x."""
        vec_latencies = _measure_latency(vector_retriever, QUERIES, runs=2)
        hyb_latencies = _measure_latency(hybrid_retriever, QUERIES, runs=2)

        vec_avg = statistics.mean(vec_latencies)
        hyb_avg = statistics.mean(hyb_latencies)

        if vec_avg > 0:
            overhead = (hyb_avg - vec_avg) / vec_avg
            assert overhead < 5.0, (
                f"Hybrid overhead {overhead:.0%} vượt giới hạn 500%"
            )

    def test_latency_report(self, vector_retriever, hybrid_retriever, capsys):
        """In báo cáo latency chi tiết (luôn pass, chỉ để xem output)."""
        vec_latencies = _measure_latency(vector_retriever, QUERIES, runs=3)
        hyb_latencies = _measure_latency(hybrid_retriever, QUERIES, runs=3)

        vec_avg = statistics.mean(vec_latencies)
        hyb_avg = statistics.mean(hyb_latencies)

        vec_sorted = sorted(vec_latencies)
        hyb_sorted = sorted(hyb_latencies)
        p95_idx = int(len(vec_sorted) * 0.95)

        print("\n" + "=" * 60)
        print("LATENCY REPORT: Vector Search vs Hybrid Search")
        print("=" * 60)
        print(f"{'Metric':<25} {'Vector':>12} {'Hybrid':>12} {'Overhead':>12}")
        print("-" * 60)
        print(f"{'Avg Latency (ms)':<25} {vec_avg:>12.2f} {hyb_avg:>12.2f} {((hyb_avg/vec_avg - 1)*100) if vec_avg > 0 else 0:>11.0f}%")
        print(f"{'P95 Latency (ms)':<25} {vec_sorted[min(p95_idx, len(vec_sorted)-1)]:>12.2f} {hyb_sorted[min(p95_idx, len(hyb_sorted)-1)]:>12.2f}")
        print(f"{'Min Latency (ms)':<25} {min(vec_latencies):>12.2f} {min(hyb_latencies):>12.2f}")
        print(f"{'Max Latency (ms)':<25} {max(vec_latencies):>12.2f} {max(hyb_latencies):>12.2f}")
        print("=" * 60)
