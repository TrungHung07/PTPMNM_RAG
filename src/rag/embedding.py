from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )