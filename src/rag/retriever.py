"""
src/rag/retriever.py — Factory functions cho các chiến lược retrieval.

Module này cung cấp 3 loại retriever có thể hoán đổi cho nhau:
  1. build_vector_retriever  — semantic search thuần túy (FAISS)
  2. build_bm25_retriever    — keyword search thuần túy (BM25)
  3. build_hybrid_retriever  — kết hợp cả hai qua Reciprocal Rank Fusion (RRF)

Thiết kế dạng factory functions (không class) để:
  - Dễ unit test từng hàm độc lập với mock/stub
  - Pipeline không phụ thuộc vào implementation cụ thể
  - Dễ thêm retriever mới (reranker, MMR...) mà không sửa pipeline

Lưu ý: HybridRetriever được implement thủ công qua RRF vì EnsembleRetriever
của LangChain không khả dụng trong langchain>=1.0. RRF là thuật toán chuẩn
được dùng trong nhiều hệ thống search hiện đại (Elasticsearch, Vespa...).
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


def build_vector_retriever(vectorstore, k: int = 4) -> BaseRetriever:
    """
    Tạo retriever dựa trên vector similarity search (FAISS).

    Phù hợp với câu hỏi mang tính ngữ nghĩa, paraphrase,
    dạng "ý của đoạn này là gì?", "tóm tắt phần nói về X".

    Hạn chế: dễ bỏ sót khi câu hỏi dùng thuật ngữ chính xác (mã số, tên riêng,
    ký hiệu kỹ thuật) vì embedding có thể ánh xạ chúng sang vector gần nhau.

    Args:
        vectorstore: FAISS vector store đã được index.
        k: Số chunk tối đa trả về (mặc định: 4).

    Returns:
        BaseRetriever — LangChain retriever sẵn sàng gọi .invoke(query).
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_bm25_retriever(chunks: list[Document], k: int = 4) -> BaseRetriever:
    """
    Tạo retriever dựa trên thuật toán BM25 (keyword-based ranking).

    BM25 (Best Match 25) là thuật toán TF-IDF nâng cao. Hoạt động tốt với
    câu hỏi chứa từ khóa chính xác, thuật ngữ chuyên ngành, mã số, tên riêng.

    Yêu cầu: thư viện `rank_bm25` phải được cài đặt (`pip install rank_bm25`).

    Args:
        chunks: Danh sách Document (chunk văn bản đã chia). BM25 index
                sẽ được xây dựng từ page_content của mỗi chunk này.
        k: Số chunk tối đa trả về (mặc định: 4).

    Returns:
        BaseRetriever — BM25Retriever sẵn sàng gọi .invoke(query).

    Raises:
        ImportError: Nếu thư viện rank_bm25 chưa được cài đặt.
    """
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


class HybridRetriever(BaseRetriever):
    """
    Retriever kết hợp BM25 (keyword search) và FAISS (semantic search)
    sử dụng thuật toán Reciprocal Rank Fusion (RRF).

    RRF gán điểm cho mỗi document dựa trên thứ hạng của nó trong từng
    danh sách kết quả, sau đó nhân với weight tương ứng:

        score(doc) = bm25_weight * 1/(rank_bm25 + k_rrf)
                   + vector_weight * 1/(rank_vector + k_rrf)

    Ưu điểm RRF so với score-based fusion:
    - Không cần normalize điểm (BM25 và cosine similarity có scale khác nhau)
    - Robust với outliers (thứ hạng ít bị ảnh hưởng bởi distribution của điểm)
    - Đơn giản, không cần tham số phức tạp

    Attributes:
        bm25_retriever: BM25 retriever đã được khởi tạo.
        vector_retriever: FAISS vector retriever đã được khởi tạo.
        bm25_weight: Trọng số ảnh hưởng của BM25, [0.0, 1.0].
        k_rrf: Hằng số RRF (thường là 60), tránh thứ hạng đầu được score quá cao.
        top_k: Số document cuối cùng trả về sau fusion.
    """
    bm25_retriever: BaseRetriever
    vector_retriever: BaseRetriever
    bm25_weight: float = 0.5
    k_rrf: int = 60
    top_k: int = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Thực hiện hybrid retrieval bằng Reciprocal Rank Fusion.

        Quy trình:
        1. Chạy song song BM25 và vector retriever với cùng query.
        2. Gán RRF score cho mỗi document dựa trên thứ hạng trong từng danh sách.
        3. Cộng điểm có trọng số, sắp xếp giảm dần, lấy top_k documents.

        Document được nhận diện unique bằng page_content (không phải object identity).

        Args:
            query: Câu hỏi/từ khóa tìm kiếm.
            run_manager: LangChain callback manager (required by interface).

        Returns:
            Danh sách Document đã được rank tổng hợp, tối đa top_k phần tử.
        """
        # ── Bước 1: Retrieve từ từng source ──────────────────────────────────
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        # ── Bước 2: Gán RRF score (dùng content làm key vì Document không hashable) ──
        scores: dict[str, float] = {}
        doc_lookup: dict[str, Document] = {}  # Lưu Document gốc để trả về

        vector_weight = 1.0 - self.bm25_weight

        # Tính RRF score từ BM25 results
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            rrf_score = self.bm25_weight * (1.0 / (rank + self.k_rrf))
            scores[key] = scores.get(key, 0.0) + rrf_score
            doc_lookup[key] = doc

        # Tính RRF score từ vector results, cộng dồn nếu đã có
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            rrf_score = vector_weight * (1.0 / (rank + self.k_rrf))
            scores[key] = scores.get(key, 0.0) + rrf_score
            doc_lookup[key] = doc

        # ── Bước 3: Sắp xếp theo điểm tổng và lấy top_k ─────────────────────
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_lookup[key] for key in sorted_keys[: self.top_k]]


def build_hybrid_retriever(
    vectorstore,
    chunks: list[Document],
    k: int = 4,
    bm25_weight: float = 0.5,
) -> HybridRetriever:
    """
    Tạo HybridRetriever kết hợp BM25 (keyword) và FAISS (semantic) qua RRF.

    Sơ đồ hoạt động:
        query ──► BM25 Retriever  ──► [doc_a rank 1, doc_b rank 2, ...]  (weight: bm25_weight)
              └──► FAISS Retriever ──► [doc_b rank 1, doc_c rank 2, ...]  (weight: 1-bm25_weight)
                                            │
                               Reciprocal Rank Fusion (RRF)
                                            │
                              [doc_b (top), doc_a, doc_c, ...]

    Args:
        vectorstore: FAISS vector store đã được index.
        chunks: Danh sách Document gốc để xây dựng BM25 index.
                Phải là CÙNG danh sách chunk đã đưa vào vectorstore.
        k: Số chunk mỗi retriever lấy ra trước khi fusion (mặc định: 4).
        bm25_weight: Trọng số BM25 trong khoảng [0.0, 1.0].
                     - 0.5: chia đôi (mặc định)
                     - 0.8: ưu tiên keyword match hơn
                     - 0.2: ưu tiên semantic search hơn

    Returns:
        HybridRetriever — BaseRetriever sẵn sàng gọi .invoke(query).

    Raises:
        ValueError: Nếu bm25_weight nằm ngoài khoảng [0.0, 1.0].
        ImportError: Nếu thư viện rank_bm25 chưa được cài đặt.
    """
    if not (0.0 <= bm25_weight <= 1.0):
        raise ValueError(
            f"bm25_weight phải trong khoảng [0.0, 1.0], nhận được: {bm25_weight}"
        )

    bm25 = build_bm25_retriever(chunks, k=k)
    vector = build_vector_retriever(vectorstore, k=k)

    return HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        bm25_weight=bm25_weight,
        top_k=k,
    )


__all__ = [
    "build_vector_retriever",
    "build_bm25_retriever",
    "build_hybrid_retriever",
    "HybridRetriever",
]
