# SmartDoc AI - Advanced Conversational RAG Platform

SmartDoc AI is a sophisticated Retrieval-Augmented Generation (RAG) platform designed to transform static documents into interactive, context-aware knowledge bases. Built with a focus on precision and user experience, it features a modern, NotebookLM-inspired interface and high-performance retrieval strategies.

---

## Key Features

### 1. Dual RAG Engines
- **Standard RAG**: Utilizes state-of-the-art vector and keyword search for rapid, accurate information retrieval.
- **Graph RAG**: Leverages Knowledge Graphs to understand complex relationships across documents, providing deeper insights and better reasoning.

### 2. Advanced Retrieval Strategies
- **Hybrid Search**: Seamlessly combines **Vector Search** (semantic meaning) with **BM25** (keyword matching) with adjustable weights.
- **PhoRanker Reranking**: Integrates a Cross-Encoder reranker to filter and prioritize the most relevant document chunks, ensuring high-fidelity citations.

### 3. Real-time Evaluation & Comparison
- **Strategy Comparison**: Side-by-side mode to evaluate Vector Search vs. Hybrid Search.
- **Engine Comparison**: Compare results from Standard RAG and Graph RAG in a dual-column layout to find the best answer.

### 4. Premium User Experience
- **NotebookLM-inspired UI**: A clean, minimalist interface using Google Sans typography and a logical two-column layout.
- **Interactive Citations**: Every response includes clickable sources with keyword highlighting, directly linking AI answers to ground truth.
- **Persistent Sessions**: Full chat history and document management powered by PostgreSQL.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS for a premium aesthetic.
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) for a high-performance asynchronous API.
- **LLM Engine**: [Ollama](https://ollama.com/) (running `qwen2.5:3b` by default).
- **Database**: [PostgreSQL](https://www.postgresql.org/) with `pgvector` for vector and metadata storage.
- **Reranker**: [PhoRanker](https://github.com/VinAIResearch/PhoRanker) (Cross-Encoder).
- **Deployment**: [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/).

---

## 🚀 Getting Started

The easiest way to run SmartDoc AI is using Docker Compose.

### Quick Start
1. **Prepare Environment**:
   ```bash
   cp .env.example .env
   ```
2. **Launch Services**:
   ```bash
   docker compose up -d
   ```
3. **Pull the LLM Model**:
   ```bash
   docker exec -it ptpmnm_ollama ollama pull qwen2.5:3b
   ```
4. **Access the App**:
   - Frontend: `http://localhost:8501` (Streamlit)
   - API Docs: `http://localhost:8000/docs` (FastAPI)

> For detailed local setup instructions and troubleshooting, see [SETUP.md](SETUP.md).

---

## 📁 Project Structure

```text
PTPMNM_RAG/
├── app.py              # Backend API (FastAPI)
├── app_fe.py           # Frontend Application (Streamlit)
├── src/                # Core logic
│   ├── rag/            # RAG pipelines, retrieval, and LLM 
|   ├── graphrag/       # Graph RAG pipeline
│   ├── database.py     # Database connections and queries
│   └── history/        # Chat session management
├── db/                 # Database migrations and seeds
├── docs/               # Feature documentation
└── docker-compose.yml  # Orchestration configuration
```

---

## 📄 License

This project is part of the Open Source Software Development course. Refer to the documentation for usage details.
