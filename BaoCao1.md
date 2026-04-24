# BÁO CÁO HỆ THỐNG RAG HỖ TRỢ TRÍCH DẪN NGUỒN VÀ HYBRID SEARCH

## I. GIỚI THIỆU

### A. Bối cảnh đề tài

Trong bối cảnh tài liệu số ngày càng tăng về số lượng và độ phức tạp, nhu cầu truy vấn thông tin chính xác, có thể kiểm chứng ngày càng quan trọng. Các mô hình ngôn ngữ lớn có khả năng sinh văn bản tự nhiên tốt, nhưng nếu không được cung cấp ngữ cảnh nguồn thì dễ phát sinh hiện tượng trả lời sai sự thật hoặc thiếu căn cứ.

Hệ thống Retrieval-Augmented Generation (RAG) được xây dựng để kết hợp truy xuất thông tin và sinh câu trả lời, từ đó đảm bảo câu trả lời bám sát tài liệu gốc. Dự án triển khai theo định hướng chạy cục bộ, sử dụng công nghệ mã nguồn mở, đồng thời bổ sung các năng lực thực tiễn như multi-file retrieval, hybrid search, re-ranking và theo vết citation.

> [PLACEHOLDER-HINH-I-1] Minh họa bài toán hỏi đáp trên tài liệu nội bộ doanh nghiệp.

### B. Mục tiêu dự án

Hệ thống đặt ra các mục tiêu chính như sau:

- Xây dựng ứng dụng hỏi đáp tài liệu theo hướng RAG, hỗ trợ nạp tài liệu định dạng PDF và DOCX.
- Tạo pipeline đầy đủ từ parsing, chunking, embedding, indexing đến retrieval và answer generation.
- Tích hợp vector search (FAISS), keyword search (BM25) và cơ chế kết hợp hybrid theo RRF.
- Cung cấp citation rõ ràng để truy vết nội dung trả lời về trang/đoạn/chunk.
- Lưu lịch sử hội thoại theo session vào PostgreSQL, hỗ trợ restore sau khi khởi động lại dịch vụ.
- Bổ sung chế độ so sánh vector và hybrid để đánh giá chất lượng retrieval.

### C. Phạm vi và bài toán

Phạm vi báo cáo tập trung vào thiết kế và triển khai hệ thống backend RAG cùng quy trình kiểm thử. Hệ thống ưu tiên hỗ trợ triển khai local hoặc Docker, cho phép mở rộng về mô hình, tham số truy xuất và chiến lược tối ưu độ chính xác.

Bài toán cần giải quyết là: với một hoặc nhiều tài liệu được nạp vào phiên làm việc, hệ thống phải trả lời câu hỏi đúng ngữ cảnh, có nguồn tham chiếu rõ ràng và giảm thiểu nhiễu trong retrieval.

## II. CƠ SỞ LÝ THUYẾT

### A. RAG (Retrieval-Augmented Generation)

RAG là mô hình kết hợp hai thành phần cốt lõi:

- Retrieval: truy xuất các đoạn văn liên quan từ kho tri thức.
- Generation: dùng mô hình ngôn ngữ để tổng hợp câu trả lời dựa trên ngữ cảnh đã truy xuất.

Biểu diễn khái quát:

RAG(x) = Generator(x, Retriever(x))

Trong đó x là câu hỏi đầu vào, Retriever(x) trả về các đoạn văn ứng viên, và Generator tổng hợp câu trả lời có điều kiện theo ngữ cảnh đó.

### B. Text Embedding và Vector Database (FAISS)

Embedding chuyển văn bản về vector số trong không gian nhiều chiều để phục vụ truy vấn ngữ nghĩa. Dự án dùng mô hình sentence-transformers để tạo embedding và lưu index bằng FAISS.

```python
# src/rag/embedding.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
```

```python
# src/rag/vectorstore.py
from langchain_community.vectorstores import FAISS


def create_vectorstore(documents, embedding):
    return FAISS.from_documents(documents, embedding)
```

Mô hình trên đạt cân bằng tốt giữa tốc độ và chất lượng truy xuất cho các bài toán RAG cỡ vừa.

### C. Large Language Models (Qwen2.5, Ollama)

Hệ thống sử dụng Ollama để phục vụ suy luận cục bộ và gọi mô hình Qwen2.5.

```python
# src/rag/llm.py
import os
from langchain_community.llms import Ollama
from functools import lru_cache


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@lru_cache(maxsize=1)
def get_llm():
    return Ollama(model="qwen2.5:3b", base_url=OLLAMA_BASE_URL, temperature=0)
```

Thiết kế gọi local giúp tăng quyền kiểm soát dữ liệu và giảm phụ thuộc dịch vụ bên ngoài.

### D. LangChain Framework

LangChain được dùng làm lớp tích hợp các thành phần: Document abstraction, retriever interface, vector store và orchestration pipeline. Thiết kế hiện tại tách module rõ ràng để dễ bảo trì và mở rộng.

### E. Hybrid Search, RRF, Reranking và Citation Tracking

Hệ thống kết hợp BM25 và vector search bằng Reciprocal Rank Fusion (RRF):

score(d) = w_bm25 * 1 / (rank_bm25(d) + k_rrf) + (1 - w_bm25) * 1 / (rank_vector(d) + k_rrf)

Trong đó w_bm25 thuộc đoạn [0.0, 1.0]. Cách kết hợp theo rank giúp ổn định hơn khi hai nguồn điểm có thang đo khác nhau.

```python
# src/rag/retriever.py (trích)
for rank, doc in enumerate(bm25_docs):
    key = doc.page_content
    rrf_score = self.bm25_weight * (1.0 / (rank + self.k_rrf))
    scores[key] = scores.get(key, 0.0) + rrf_score

for rank, doc in enumerate(vector_docs):
    key = doc.page_content
    rrf_score = vector_weight * (1.0 / (rank + self.k_rrf))
    scores[key] = scores.get(key, 0.0) + rrf_score
```

Reranking dùng cross-encoder để lọc và sắp xếp lại các đoạn retrieved theo độ liên quan ngữ nghĩa query-chunk.

```python
# src/rag/rerank.py (trích)
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
cutoff = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))
ranked = [(d, s) for d, s in ranked if s >= cutoff]
ranked = ranked[: min(top_k, len(ranked))]
```

Citation được tạo từ tập document cuối cùng sau retrieval/rerank và giữ nguyên metadata để truy vết.

```python
# src/rag/pipeline.py (trích)
def _build_citation_list(docs, scores=None):
    citations = []
    for i, doc in enumerate(docs):
        score = float(scores[i]) if scores and i < len(scores) else None
        citations.append(CitationSource(content=doc.page_content, metadata=doc.metadata, score=score))
    return citations
```

> [PLACEHOLDER-HINH-II-1] Sơ đồ tổng quan retrieval vector, BM25, hybrid và rerank.

## III. THIẾT KẾ HỆ THỐNG

### A. Kiến trúc tổng quan

Kiến trúc hệ thống được phân thành các lớp chính:

- Ingestion layer: nhận file, parse và chuẩn hóa metadata.
- Retrieval layer: FAISS, BM25, hybrid RRF, rerank.
- Generation layer: xây prompt và gọi LLM qua Ollama.
- Persistence layer: PostgreSQL lưu session, documents, messages, citations.

> [PLACEHOLDER-HINH-III-A] Sơ đồ kiến trúc nhiều lớp của hệ thống.

### B. Luồng xử lý tài liệu

Quy trình ingest tài liệu:

1. Nhận file upload qua API.
2. Lưu file vật lý vào thư mục data.
3. Parse tài liệu thành danh sách Document.
4. Chunking có overlap và bảo toàn metadata.
5. Tạo embedding và FAISS index, lưu vào INDEX_DB theo session/file.

```python
# app.py (trích)
file_path = Path(f"data/{file_id}_{file.filename}")
documents = load_documents(file_path)
index = build_index(documents, chunk_size=chunk_size, overlap=chunk_overlap)
INDEX_DB[session_id][file_id] = index
```

### C. Luồng xử lý câu hỏi

Quy trình xử lý câu hỏi gồm các bước:

1. Nhận request chứa session_id, file_ids, question và search_mode.
2. Nếu index bị mất trong memory thì tự restore từ file gốc.
3. Gộp index theo danh sách file_ids.
4. Chọn retriever theo mode (vector hoặc hybrid).
5. Tùy chọn rerank và lọc theo ngưỡng.
6. Build prompt, gọi LLM, tạo citations.
7. Lưu lịch sử hội thoại và citation vào DB.

```python
# src/rag/pipeline.py (trích)
if search_mode == "hybrid":
    retriever = build_hybrid_retriever(
        vectorstore=index.vectorstore,
        chunks=index.chunks,
        k=retrieve_candidates,
        bm25_weight=bm25_weight,
    )
else:
    retriever = build_vector_retriever(index.vectorstore, k=retrieve_candidates)

docs = retriever.invoke(question)
```

### D. Thiết kế các thành phần mã nguồn

#### Parser PDF/DOCX

```python
# src/parsers/pdf_parser.py (trích)
documents.append(Document(
    page_content=page_text,
    metadata={"page": i + 1, "source": path.name}
))
```

```python
# src/parsers/docx_parser.py (trích)
documents.append(Document(
    page_content=text,
    metadata={"paragraph": paragraph_index, "source": path.name}
))
```

#### Chunking và bảo toàn metadata

```python
# src/chunking/text_chunker.py (trích)
chunk_metadata = {**doc.metadata, "chunk_index": chunk_index}
result.append(Document(
    page_content=chunk_text_content,
    metadata=chunk_metadata,
))
```

#### Lưu lịch sử hội thoại và citation

```python
# src/database.py (trích)
await conn.execute(
    """
    INSERT INTO messages (session_id, question, answer, file_ids, search_mode, citations)
    VALUES ($1::uuid, $2, $3, $4::uuid[], $5, $6::jsonb)
    """,
    session_id, question, answer, uuid_list, search_mode, citations_json,
)
```

### E. Prompt Engineering và quản lý hội thoại

Prompt được thiết kế để ép mô hình bám context tài liệu và trả lời cùng ngôn ngữ truy vấn.

```python
# src/rag/pipeline.py (trích)
def _build_prompt(context: str, question: str, history_text: str) -> str:
    history_section = f"""
Lịch sử hội thoại trước đó (để hiểu ngữ cảnh follow-up):
{history_text}
""" if history_text else ""

    return f"""
BẠN LÀ MỘT TRỢ LÝ AI CHUYÊN NGHIỆP. BẠN TUYỆT ĐỐI CHỈ ĐƯỢC DÙNG TIẾNG VIỆT HOẶC TIẾNG ANH.
...
Context từ tài liệu:
{context}

{history_section}

Câu hỏi hiện tại: {question}
TRẢ LỜI (CHỈ DÙNG TIẾNG VIỆT HOẶC TIẾNG ANH):
""".strip()
```

> [PLACEHOLDER-HINH-III-B] Sequence diagram luồng upload và indexing.
>
> [PLACEHOLDER-HINH-III-C] Sequence diagram luồng ask/compare.
>
> [PLACEHOLDER-HINH-III-D] ERD cho sessions, documents, messages.

## IV. TRIỂN KHAI HỆ THỐNG

### A. Công nghệ sử dụng

- Backend API: FastAPI.
- Frontend: Streamlit.
- Retrieval stack: LangChain + FAISS + BM25 custom hybrid.
- Rerank: sentence-transformers CrossEncoder.
- LLM runtime: Ollama.
- CSDL: PostgreSQL + asyncpg.
- Triển khai: Docker Compose hoặc local environment.

### B. Cài đặt môi trường (local, Docker)

Quy trình cơ bản:

1. Tạo môi trường ảo và cài dependencies.
2. Khởi động PostgreSQL và Ollama.
3. Pull mô hình Qwen2.5.
4. Chạy FastAPI và Streamlit.

Lệnh mẫu:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

docker compose up -d
docker exec -it ptpmnm_ollama ollama pull qwen2.5:3b
uvicorn app:app --reload
streamlit run app_fe.py
```

### C. Cấu hình mô hình và tham số

Các tham số chính có thể điều chỉnh gồm:

- chunk_size, chunk_overlap tại thời điểm upload.
- search_mode và bm25_weight khi đặt câu hỏi.
- RERANK_ENABLED, RERANK_TOP_K, RERANK_SCORE_THRESHOLD cho bước rerank.
- HISTORY_WINDOW và RETRIEVER_TOP_K trong pipeline.

### D. Cơ sở dữ liệu và quản lý lịch sử hội thoại

Schema được mở rộng theo từng migration để hỗ trợ bối cảnh multi-file và audit retrieval.

```sql
-- db/migrations/002_messages_file_ids.sql
ALTER TABLE "public"."messages"
  ADD COLUMN IF NOT EXISTS "file_ids" uuid[] NOT NULL DEFAULT '{}';
```

```sql
-- db/migrations/003_messages_search_mode_citations.sql
ALTER TABLE "public"."messages"
  ADD COLUMN IF NOT EXISTS "search_mode" text NOT NULL DEFAULT 'unknown';

ALTER TABLE "public"."messages"
  ADD COLUMN IF NOT EXISTS "citations" jsonb NOT NULL DEFAULT '[]'::jsonb;
```

Lọc lịch sử gần đây sử dụng điều kiện giao mảng file_ids để giữ đúng ngữ cảnh:

```sql
WHERE session_id = $1::uuid
  AND file_ids && $2::uuid[]
```

### E. API chính và giao diện sử dụng

Các endpoint chính:

- POST /upload: nạp tài liệu, tạo index theo session.
- POST /ask: hỏi đáp theo vector/hybrid.
- POST /compare: chạy song song vector và hybrid để so sánh.
- POST /sessions/{session_id}/restore: khôi phục index từ dữ liệu lưu trên disk.
- GET /history: lấy danh sách session và lịch sử hội thoại.

```python
# app.py (trích)
await append_message(
    req.session_id,
    req.question,
    vector_result.answer,
    req.file_ids,
    search_mode="compare_vector",
    citations=[c.model_dump() for c in vector_result.citations],
)
await append_message(
    req.session_id,
    req.question,
    hybrid_result.answer,
    req.file_ids,
    search_mode="compare_hybrid",
    citations=[c.model_dump() for c in hybrid_result.citations],
)
```

> [PLACEHOLDER-HINH-IV-1] Ảnh màn hình giao diện upload và chat.
>
> [PLACEHOLDER-HINH-IV-2] Ảnh màn hình chế độ so sánh vector/hybrid.

## V. KIỂM THỬ VÀ ĐÁNH GIÁ

### A. Chiến lược kiểm thử

Hệ thống được kiểm thử theo hai lớp:

- Unit tests cho từng module lõi.
- Integration tests cho phần lưu lịch sử và tương thích dữ liệu.

Phạm vi kiểm thử bao phủ toàn bộ pipeline chính, từ parsing/chunking đến retrieval, prompt, citation và persistence.

### B. Kết quả tổng hợp theo module

Nội dung kiểm thử được tổng hợp theo đúng báo cáo kiểm thử hiện có.

| Module | File kiểm thử | Số lượng test case | Trạng thái |
|---|---|---:|---|
| Chunking | test_chunker.py | 12 | Pass |
| Retriever | test_retriever.py | 8 | Pass |
| Pipeline | test_pipeline.py | 7 | Pass |
| Parsers | test_parsers.py | 4 | Pass |
| Models | test_models.py | 3 | Pass |
| Lịch sử và Database | test_history_file_ids.py, test_history_search_mode_citations.py | 9 | Pass |
| **Tổng** |  | **43** | **Pass** |

Các điểm xác nhận chính theo module:

- Chunking: kiểm tra kích thước chunk, overlap, metadata preservation và đánh chỉ mục chunk.
- Retriever: xác nhận top-k, hành vi BM25/vector/hybrid và kiểm tra trọng số bm25_weight hợp lệ.
- Pipeline: xác minh build index, xây prompt, format history và tạo citation.
- Parsers: xác minh bóc tách DOCX/PDF, bỏ qua đoạn rỗng, metadata source và paragraph/page.
- Models: xác nhận tính hợp lệ của schema và khả năng serialize response.
- History/DB: xác minh lọc theo file_ids, lưu search_mode, lưu citations JSONB, và backward compatibility.

### C. Kết quả benchmark chunking

| Chunk Size | Overlap | Avg Latency (ms) | Hit Rate (@3) | Chunk Count |
|---:|---:|---:|---:|---:|
| 500 | 50 | 35.35 | 0.75 | 6 |
| 500 | 100 | 11.03 | 1.00 | 7 |
| 500 | 200 | 13.45 | 1.00 | 9 |
| 1000 | 50 | 16.51 | 1.00 | 3 |
| 1000 | 100 | 16.51 | 1.00 | 3 |
| 1000 | 200 | 11.22 | 1.00 | 4 |
| 1500 | 50 | 16.05 | 1.00 | 2 |
| 1500 | 100 | 13.14 | 1.00 | 2 |
| 1500 | 200 | 13.25 | 1.00 | 2 |
| 2000 | 50 | 11.32 | 1.00 | 2 |
| 2000 | 100 | 16.97 | 1.00 | 2 |
| 2000 | 200 | 10.83 | 1.00 | 2 |

Nhận xét:

- Cấu hình 500/50 cho chất lượng thấp hơn (hit rate 0.75), thể hiện nguy cơ mất ngữ cảnh khi chunk nhỏ và overlap thấp.
- Hầu hết cấu hình còn lại đạt hit rate 1.00.
- Cấu hình 1000/200 hoặc 2000/200 cho cân bằng tốt giữa số lượng chunk và độ trễ.

### D. So sánh vector, hybrid, rerank

Đánh giá định tính trên hành vi hệ thống:

- Vector: phù hợp câu hỏi diễn đạt tự nhiên, paraphrase.
- Hybrid: mạnh hơn khi câu hỏi có từ khóa chính xác, tên riêng hoặc mã số.
- Rerank: cải thiện độ sạch citation nhưng tăng chi phí tính toán.

> [PLACEHOLDER-BANG-V-1] Bảng định lượng so sánh vector/hybrid/rerank trên cùng bộ câu hỏi đánh giá.
>
> [PLACEHOLDER-HINH-V-1] Biểu đồ latency cho retrieval và reranking theo từng chế độ.

### E. Hạn chế hiện tại

- Index FAISS đang quản lý in-memory, cần restore khi dịch vụ restart.
- Khi gộp nhiều file, chi phí re-embedding có thể tăng theo quy mô dữ liệu.
- Rerank cần thêm tài nguyên CPU/GPU để ổn định latency.
- Chưa có benchmark tải lớn theo nhiều người dùng đồng thời.

## VI. HƯỚNG DẪN SỬ DỤNG

### A. Quy trình nạp tài liệu

1. Gọi endpoint upload kèm danh sách file PDF/DOCX.
2. Nhận về session_id và file_id cho từng tài liệu.
3. Lưu các định danh này để sử dụng trong các lượt hỏi đáp tiếp theo.

Ví dụ:

```bash
curl -X POST "http://localhost:8000/upload?chunk_size=1000&chunk_overlap=200" \
  -F "files=@./data/tailieu1.pdf" \
  -F "files=@./data/tailieu2.docx"
```

### B. Quy trình đặt câu hỏi

1. Gửi session_id, file_ids và question vào endpoint /ask.
2. Chọn search_mode phù hợp:
   - vector khi ưu tiên semantic retrieval.
   - hybrid khi cần cân bằng semantic và keyword retrieval.
3. Nhận về answer và danh sách citations.

Ví dụ payload:

```json
{
  "session_id": "...",
  "file_ids": ["...", "..."],
  "question": "Điều khoản thanh toán được quy định như thế nào?",
  "search_mode": "hybrid",
  "bm25_weight": 0.5,
  "rerank_enabled": true,
  "rerank_threshold": 0.0
}
```

### C. Quản lý session và khôi phục dữ liệu

- Khi server restart, hệ thống có thể tự động restore index khi gọi /ask hoặc /compare.
- Có thể chủ động gọi endpoint restore để pre-warm session trước khi sử dụng.
- Lịch sử hội thoại được lưu bền vững trong PostgreSQL.

Ví dụ restore:

```bash
curl -X POST "http://localhost:8000/sessions/<session_id>/restore"
```

> [PLACEHOLDER-HINH-VI-1] Lưu đồ thao tác người dùng trên giao diện từ upload đến truy vấn.

## VII. ĐỊNH HƯỚNG PHÁT TRIỂN

### A. Cải thiện độ chính xác truy xuất

- Mở rộng bộ chỉ số đánh giá retrieval như MRR, NDCG ngoài hit rate.
- Nghiên cứu adaptive weighting cho BM25/vector theo loại câu hỏi.
- Bổ sung query rewriting/decomposition cho câu hỏi nhiều bước.

### B. Mở rộng hiệu năng và triển khai thực tế

- Lưu snapshot FAISS để giảm chi phí rebuild index.
- Tối ưu batch rerank và cache embedding.
- Bổ sung monitoring chi tiết cho retrieval_ms, rerank_ms và llm_ms.

### C. Nâng cấp tính năng sản phẩm

- Thêm bộ lọc citation theo ngưỡng độ tin cậy trên UI.
- Hỗ trợ thêm định dạng dữ liệu đầu vào như TXT, HTML, CSV.
- Bổ sung phân quyền người dùng và quản trị session theo vai trò.

## VIII. KẾT LUẬN

### A. Tổng kết kết quả đạt được

Hệ thống đã hoàn thiện các năng lực cốt lõi của một nền tảng RAG phục vụ hỏi đáp tài liệu:

- Nạp và xử lý tài liệu đa định dạng PDF/DOCX.
- Truy xuất kết hợp vector và keyword bằng hybrid RRF.
- Tạo citation có metadata để kiểm chứng câu trả lời.
- Lưu và khôi phục ngữ cảnh hội thoại theo session/file.
- Cung cấp chế độ compare để đánh giá chiến lược retrieval.
- Có bộ kiểm thử tổng thể 43 test case với trạng thái Pass.

### B. Giá trị ứng dụng

Giải pháp giúp nâng cao độ tin cậy của hệ thống hỏi đáp tài liệu nội bộ, giảm rủi ro hallucination và tạo nền tảng kỹ thuật rõ ràng cho triển khai thực tế trong môi trường doanh nghiệp hoặc học thuật.

## IX. TÀI LIỆU THAM KHẢO

1. Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020.
2. Robertson et al., Okapi at TREC-3, 1994.
3. Johnson, Douze, Jegou, Billion-scale similarity search with GPUs, 2017.
4. Reimers and Gurevych, Sentence-BERT, 2019.
5. Cormack, Clarke, Buettcher, Reciprocal Rank Fusion, SIGIR 2009.

