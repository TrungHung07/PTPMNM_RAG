from src.chunking.text_chunker import chunk_text
from src.rag.embedding import get_embedding
from src.rag.vectorstore import create_vectorstore
from src.rag.llm import get_llm

# Số lượng tin nhắn gần nhất được đưa vào ngữ cảnh hội thoại
HISTORY_WINDOW = 5


def build_vectorstore_from_text(text: str):
    chunks = chunk_text(text)
    embedding = get_embedding()
    vectorstore = create_vectorstore(chunks, embedding)
    return vectorstore


def _format_chat_history(chat_history: list) -> str:
    """Chuyển danh sách {question, answer} thành chuỗi hội thoại cho prompt."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        lines.append(f"Human: {msg['question']}")
        lines.append(f"AI: {msg['answer']}")
    return "\n".join(lines)


def ask_question(vectorstore, question: str, chat_history: list = []):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

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

    return llm.invoke(prompt)