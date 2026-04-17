from src.chunking.text_chunker import chunk_text
from src.rag.embedding import get_embedding
from src.rag.vectorstore import create_vectorstore
from src.rag.llm import get_llm


def build_vectorstore_from_text(text: str):
    chunks = chunk_text(text)
    embedding = get_embedding()
    vectorstore = create_vectorstore(chunks, embedding)
    return vectorstore


def ask_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = get_llm()

    prompt = f"""
Bạn là AI chỉ được phép trả lời dựa trên thông tin trong tài liệu dưới đây.

QUY TẮC:
- CHỈ sử dụng thông tin có trong Context.
- KHÔNG được suy đoán, KHÔNG thêm kiến thức bên ngoài.
- Nếu không tìm thấy câu trả lời trong Context, hãy trả lời:
  "Không tìm thấy thông tin trong tài liệu."


{context}

Câu hỏi: {question}
"""

    return llm.invoke(prompt)