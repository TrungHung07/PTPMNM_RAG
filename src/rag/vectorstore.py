from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_vectorstore(chunks: list[str], embedding):
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embedding)