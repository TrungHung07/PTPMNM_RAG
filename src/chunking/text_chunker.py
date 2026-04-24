from __future__ import annotations

from langchain_core.documents import Document


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    [Legacy] Chia một chuỗi văn bản thành các chunk có độ dài cố định và overlap.

    Hàm này giữ lại để tương thích ngược. Với tính năng citation, dùng
    chunk_documents() thay thế để bảo toàn metadata gốc của từng chunk.

    Args:
        text: Chuỗi văn bản cần chia.
        chunk_size: Số ký tự tối đa của mỗi chunk (mặc định: 1000).
        overlap: Số ký tự chồng lắp giữa 2 chunk liên tiếp (mặc định: 200).

    Returns:
        Danh sách các chuỗi chunk.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Document]:
    """
    Chia danh sách Document thành các chunk nhỏ hơn, bảo toàn toàn bộ metadata gốc.

    Đây là hàm cốt lõi để hỗ trợ citation tracking: mỗi chunk sinh ra sẽ giữ
    nguyên metadata của Document nguồn (page, paragraph, source...), đồng thời
    bổ sung thêm 'chunk_index' (số thứ tự chunk trong Document đó).

    Cách hoạt động:
    - Với mỗi Document, chia page_content ra thành nhiều đoạn nhỏ theo
      sliding window (chunk_size, overlap).
    - Mỗi đoạn nhỏ được đóng gói thành Document mới, kế thừa metadata gốc.

    Args:
        documents: Danh sách Document (kèm metadata) đầu vào.
        chunk_size: Số ký tự tối đa của mỗi chunk (mặc định: 1000).
        overlap: Số ký tự chồng lắp giữa 2 chunk liên tiếp (mặc định: 200).

    Returns:
        Danh sách Document đã được chia nhỏ, mỗi chunk chứa:
        - page_content: đoạn văn bản nhỏ
        - metadata: bản sao metadata gốc + thêm 'chunk_index' (0-indexed)

    Raises:
        ValueError: Nếu chunk_size <= 0 hoặc overlap < 0.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    step = max(1, chunk_size - overlap)
    result: list[Document] = []

    for doc in documents:
        text = doc.page_content
        if not text:
            continue

        chunk_index = 0
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text_content = text[start:end]

            # Kế thừa toàn bộ metadata gốc, bổ sung chunk_index để phân biệt
            # các chunk trong cùng một trang/đoạn văn
            chunk_metadata = {**doc.metadata, "chunk_index": chunk_index}

            result.append(Document(
                page_content=chunk_text_content,
                metadata=chunk_metadata,
            ))

            if end >= len(text):
                break

            start += step
            chunk_index += 1

    return result


__all__ = ["chunk_text", "chunk_documents"]
