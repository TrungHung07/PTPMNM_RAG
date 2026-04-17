"""
RAG Pipeline — Xây dựng vectorstore và xử lý Q&A có kèm citation tracking.
"""
from langchain_core.documents import Document

from src.chunking.text_chunker import chunk_documents
from src.rag.embedding import get_embedding
from src.rag.vectorstore import create_vectorstore
from src.rag.llm import get_llm
from src.models import CitationSource, AskResponse

# Số lượng tin nhắn gần nhất được đưa vào ngữ cảnh hội thoại
HISTORY_WINDOW = 5

# Số chunk được retriever lấy ra để làm context cho LLM
RETRIEVER_TOP_K = 4


def build_vectorstore_from_documents(documents: list[Document], chunk_size: int = 1000, overlap: int = 200):
    """
    Xây dựng FAISS vector store từ danh sách Document (đã có metadata).

    Quá trình:
    1. Chia nhỏ các Document thành chunk theo chunk_size và overlap.
    2. Embed các chunk bằng HuggingFace sentence-transformers.
    3. Lưu vào FAISS index (kèm toàn bộ metadata để phục vụ citation).

    Args:
        documents: Danh sách Document có metadata (page/paragraph/source...)
                   được trả về từ pdf_parser hoặc docx_parser.
        chunk_size: Số ký tự tối đa của mỗi chunk (mặc định: 1000).
        overlap: Số ký tự overlap giữa các chunk liên tiếp (mặc định: 200).

    Returns:
        FAISS vector store đã được index, sẵn sàng cho similarity search.
    """
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    embedding = get_embedding()
    return create_vectorstore(chunks, embedding)


def build_vectorstore_from_text(text: str):
    """
    [Legacy] Xây dựng FAISS vector store từ một chuỗi văn bản thuần túy.

    Hàm này giữ lại để tương thích ngược với code cũ chưa hỗ trợ metadata.
    Các Document tạo ra sẽ không có metadata page/source nên citations
    sẽ không có thông tin vị trí.

    Args:
        text: Chuỗi văn bản đầy đủ cần đưa vào vector store.

    Returns:
        FAISS vector store đã được index.
    """
    doc = Document(page_content=text, metadata={"source": "unknown"})
    return build_vectorstore_from_documents([doc])


def _format_chat_history(chat_history: list) -> str:
    """
    Chuyển danh sách lịch sử hội thoại [{question, answer}] thành chuỗi
    plain text để đưa vào prompt cho LLM.

    Args:
        chat_history: Danh sách dict {'question': str, 'answer': str}.

    Returns:
        Chuỗi hội thoại định dạng "Human: ...\nAI: ...", hoặc "" nếu rỗng.
    """
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        lines.append(f"Human: {msg['question']}")
        lines.append(f"AI: {msg['answer']}")
    return "\n".join(lines)


def _build_citation_list(docs: list[Document]) -> list[CitationSource]:
    """
    Chuyển danh sách Document được retriever trả về thành danh sách CitationSource.

    Mỗi CitationSource chứa:
    - content: đoạn văn bản gốc của chunk
    - metadata: thông tin vị trí (page, paragraph, source, chunk_index...)

    Args:
        docs: Danh sách Document từ kết quả retrieval FAISS.

    Returns:
        Danh sách CitationSource dùng để trả về trong API response.
    """
    return [
        CitationSource(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]


def ask_question(
    vectorstore,
    question: str,
    chat_history: list = [],
) -> AskResponse:
    """
    Truy vấn RAG: tìm các đoạn văn liên quan, tạo câu trả lời và trả về citations.

    Quy trình:
    1. Dùng FAISS retriever tìm TOP_K chunk gần nhất với câu hỏi.
    2. Xây dựng context prompt từ các chunk và lịch sử hội thoại.
    3. Gọi Ollama LLM để sinh câu trả lời.
    4. Đóng gói answer + danh sách citations (chunk gốc và metadata vị trí).

    Args:
        vectorstore: FAISS vector store đã được index cho file tài liệu.
        question: Câu hỏi của người dùng cần trả lời.
        chat_history: Lịch sử hội thoại gần đây, mỗi item là {'question', 'answer'}.

    Returns:
        AskResponse chứa:
        - question: câu hỏi gốc
        - answer: câu trả lời của AI
        - citations: danh sách CitationSource (content + metadata)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    # Lấy các chunk liên quan — đây cũng chính là nguồn citation
    docs = retriever.invoke(question)

    # Ghép nội dung các chunk thành context cho prompt
    context = "\n\n".join([doc.page_content for doc in docs])

    # Chỉ lấy HISTORY_WINDOW tin nhắn gần nhất để tránh prompt quá dài
    recent_history = chat_history[-HISTORY_WINDOW:]
    history_text = _format_chat_history(recent_history)

    history_section = f"""
Lịch sử hội thoại trước đó (để hiểu ngữ cảnh follow-up):
{history_text}
""" if history_text else ""

    llm = get_llm()

    prompt = f"""
Bạn là AI chỉ được phép trả lời dựa trên thông tin trong tài liệu dưới đây.

QUY TẮC:
- CHỈ sử dụng thông tin có trong Context.
- KHÔNG được suy đoán, KHÔNG thêm kiến thức bên ngoài.
- Nếu câu hỏi là follow-up (ví dụ: "giải thích thêm", "ý đó là gì"), hãy dùng Lịch sử hội thoại để hiểu ngữ cảnh.
- Nếu không tìm thấy câu trả lời trong Context, hãy trả lời:
  "Không tìm thấy thông tin trong tài liệu."

Context từ tài liệu:
{context}
{history_section}
Câu hỏi hiện tại: {question}
"""

    answer = llm.invoke(prompt)

    # Xây dựng danh sách citations từ các chunk đã retrieve
    citations = _build_citation_list(docs)

    return AskResponse(question=question, answer=answer, citations=citations)


__all__ = ["build_vectorstore_from_documents", "build_vectorstore_from_text", "ask_question"]