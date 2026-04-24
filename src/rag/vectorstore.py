from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_vectorstore(documents: list[Document], embedding):
    """
    Tạo FAISS vector store từ danh sách Document (đã có metadata).

    Hàm này nhận trực tiếp list[Document] thay vì list[str] để bảo toàn
    metadata (page, paragraph, source...) đã được gắn sẵn từ bước chunking.
    Toàn bộ metadata này được FAISS lưu lại và có thể truy xuất khi retrieval.

    Args:
        documents: Danh sách Document kèm metadata cần embedding và lưu trữ.
        embedding: Embedding model (ví dụ: HuggingFaceEmbeddings).

    Returns:
        FAISS vector store đã được index, sẵn sàng cho similarity search.
    """
    return FAISS.from_documents(documents, embedding)