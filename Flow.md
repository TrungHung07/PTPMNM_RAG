# Flow hiện tại của ChatBox Local RAG

Tài liệu này mô tả **luồng chạy hiện tại** của project (theo code trong `src/chatbox/`), gồm 2 đường chính:

- **CLI**: import tài liệu và query demo local.
- **HTTP API (FastAPI routers)**: cung cấp endpoint ingest và query/streaming (hiện repo mới có router/entrypoints, chưa thấy `FastAPI()` app wiring).

## Tổng quan kiến trúc module

- **Ingestion**: `chatbox.ingestion.*` + `chatbox.ingestion.parsers.*`
- **Chunking**: `chatbox.chunking.*`
- **Storage (local-first)**: `chatbox.storage.sqlite_metadata.SqliteMetadataStore` (SQLite tại `.chatbox/chatbox.db`)
- **Retrieval**
  - **RAG (vector)**: `chatbox.rag.*` (embedding stub + in-memory vector store)
  - **GraphRAG**: `chatbox.graphrag.*` (in-memory graph store + entity/relation extractors)
- **Orchestration**
  - **Parallel retrieval + merge**: `chatbox.orchestration.parallel_retrieval.ParallelRetriever` + `chatbox.orchestration.merger`
  - **Generation + streaming/fallback**: `chatbox.orchestration.llm_response.LlmResponseOrchestrator` + providers `ollama_adapter` / `vllm_adapter`
- **App bootstrap**: `chatbox.app.bootstrap.create_app_container` (config logging + khởi tạo SQLite store + metrics/tracing)

## Flow 1: Import/Ingestion (CLI + API ingest)

### Ý nghĩa

Nhập một file PDF/DOCX, parse thành text, normalize, chunk theo token window, sau đó **persist metadata + chunks vào SQLite**.

### CLI entrypoint

- `python -m chatbox.app.cli.import_cmd run --file-path <path> --file-type pdf|docx [--sqlite-path .chatbox/chatbox.db]`
- Code: `src/chatbox/app/cli/import_cmd.py`

### Pipeline chi tiết

```mermaid
flowchart TD
  A[CLI: import_cmd.run_import] --> B[create_app_container(sqlite_path)]
  B --> C[SqliteMetadataStore(.chatbox/chatbox.db)]
  A --> D[IngestionCoordinator(metadata_store, parsers)]
  D --> E[Parser.parse(file_path)]
  E --> F[normalize_text(parsed.text)]
  F --> G[chunk_text(normalized_text, document_id)]
  G --> H[metadata_store.save_document(Document)]
  G --> I[metadata_store.save_chunks(list[Chunk])]
  H --> J[IngestResponse(document_id, chunk_count, ingest_status)]
  I --> J
```

### Thành phần tham gia (theo file)

- **Coordinator**: `src/chatbox/ingestion/coordinator.py`
  - Chọn parser theo `file_type`
  - Tạo `Document` (id dạng `doc-<uuid>`)
  - Chunk text, lưu `documents` + `chunks`
- **Normalize**: `src/chatbox/ingestion/normalizer.py`
  - Chuẩn hoá newline, rút gọn chuỗi dòng trống
- **Chunking**: `src/chatbox/chunking/chunker.py`
  - Tokenize → chia cửa sổ `max_tokens=300`, overlap `overlap_tokens=30`
  - `chunk_id` ổn định theo hash \(document_id|chunk_order|text\)
- **Storage SQLite**: `src/chatbox/storage/sqlite_metadata.py`
  - Tự tạo schema `documents` và `chunks` nếu chưa có
  - DB nằm dưới `.chatbox/` (đã git-ignore theo README)

### API ingest (router)

- Endpoint: `POST /v1/documents:ingest`
- Router: `src/chatbox/app/api/documents.py`
  - Nhận `IngestRequest` \(`source_uri`, `file_type`\) → gọi `IngestionCoordinator.ingest(...)`

## Flow 2: Query / Retrieval / Generation (CLI + API query)

### Ý nghĩa

Nhận câu hỏi, tạo **retrieval context** theo mode:

- **rag**: chỉ vector retrieval
- **hybrid**: chạy **vector + graph** song song → merge evidence

Sau đó (nếu có orchestrator) **generate câu trả lời** với **primary provider**, fallback sang **secondary provider** nếu lỗi; hỗ trợ **streaming text/plain**.

### API query routers

- `POST /v1/query/context`: chỉ trả context
- `POST /v1/query`: trả answer (stream hoặc JSON)
- Router: `src/chatbox/app/api/query.py`

### Pipeline chi tiết (API)

```mermaid
flowchart TD
  A[POST /v1/query or /v1/query/context] --> B[create_query_router(...)]
  B --> C{request.mode}
  C -->|rag| D[rag_retriever.retrieve(query_text, top_k)]
  C -->|hybrid| E[ParallelRetriever.retrieve async]
  E --> E1[rag retrieve in thread]
  E --> E2[graph retrieve in thread]
  E1 --> F[merge_hybrid_context]
  E2 --> F
  D --> G[merge_rag_context]
  F --> H[RetrievalContext]
  G --> H
  H --> I{endpoint}
  I -->|/query/context| J[QueryContextResponse]
  I -->|/query| K{llm_orchestrator configured?}
  K -->|no| L["No LLM provider configured..."]
  K -->|yes| M[llm_orchestrator.generate(prompt)]
  M --> N{stream?}
  L --> N
  N -->|true| O[StreamingResponse text/plain]
  N -->|false| P[QueryAnswerResponse JSON]
```

### Retrieval components (theo file)

- **RAG retriever**: `src/chatbox/rag/retriever.py`
  - Embed query bằng `EmbeddingService` → search trong `InMemoryVectorStore`
- **Embedding stub**: `src/chatbox/rag/embeddings.py`
  - Embedding deterministic bằng hash token (phục vụ offline tests/demo)
- **Vector store + indexer**: `src/chatbox/rag/indexer.py`
  - `InMemoryVectorStore` lưu vectors, search cosine similarity
  - `RagIndexer.index_chunks(...)` để upsert embeddings và attach text
- **Graph retrieval**: `src/chatbox/graphrag/builder.py` + `src/chatbox/graphrag/retriever.py`
  - `GraphBuilder.build_from_chunks` trích entities/relations → upsert graph store
  - `GraphRetriever.retrieve` trích entities từ query → search_paths trong store
- **Parallel retrieval + degrade**: `src/chatbox/orchestration/parallel_retrieval.py`
  - Chạy 2 nhánh bằng `asyncio.to_thread` và `gather(return_exceptions=True)`
  - `degraded_mode`: `rag_only` / `graph_only` / `retrieval_unavailable`
- **Merge context**: `src/chatbox/orchestration/merger.py`
  - `merge_rag_context` và `merge_hybrid_context` tạo `RetrievalContext`

### Generation + streaming/fallback (theo file)

- **Orchestrator**: `src/chatbox/orchestration/llm_response.py`
  - `_build_prompt`: lấy top evidence (tối đa 5) làm prompt
  - `generate`: gọi `primary_provider.generate(...)`, fallback qua secondary nếu exception
  - `stream_text`: chia text theo chunk để stream
- **Providers (stub)**:
  - `src/chatbox/orchestration/providers/ollama_adapter.py`
  - `src/chatbox/orchestration/providers/vllm_adapter.py`
  - Hiện tại đều trả về “preview” prompt (phục vụ demo/offline)

## Flow 3: Query demo qua CLI (lưu ý khác với API)

CLI query hiện là **demo in-memory** (không đọc từ SQLite/chunks đã import). Nó tự seed 1 chunk mẫu vào cả vector index và graph:

- Entry: `src/chatbox/app/cli/query_cmd.py`
  - Tạo `EmbeddingService` + `InMemoryVectorStore` + `RagIndexer/RagRetriever`
  - Nếu `mode == "hybrid"`: tạo `InMemoryGraphStore` + `GraphBuilder` + `GraphRetriever`
  - Seed chunk `("seed-1", "ChatBox supports local-first retrieval")`
  - Chạy `ParallelRetriever.retrieve(...)`
  - Generate bằng `LlmResponseOrchestrator(primary=OllamaAdapter, secondary=VllmAdapter)`
  - Nếu `--stream true`: in ra từng đoạn bằng `orchestrator.stream_text(...)`

## Local-first state

- **SQLite metadata DB**: mặc định `.chatbox/chatbox.db`
- **Tạo schema tự động** khi khởi tạo `SqliteMetadataStore`
- `README.md` ghi rõ `.chatbox/` là local state và đã được git-ignore

## Ghi chú hiện trạng (dựa trên code hiện tại)

- **API layer**: repo có router cho FastAPI (`create_query_router`, `create_documents_router`) nhưng chưa thấy file khởi tạo `FastAPI()` và `include_router(...)` trong `src/chatbox/app/`.
- **Retrieval khi query**: hiện **không** có đoạn code nối từ SQLite chunks → index vector/graph cho query API; retrievers cần được inject từ nơi “wiring” (chưa có trong repo).
- **LLM providers**: đang là adapter stub (preview prompt), mục tiêu là demo/offline/test.

