# RAG Re-ranking (Cross-Encoder) — Kế hoạch triển khai

## Mục tiêu
- Thêm **bước re-ranking sau retrieval**.
- Dùng **cross-encoder** để chấm lại độ liên quan của các chunk đã retrieve.
- Có **so sánh công bằng** với approach hiện tại dùng **bi-encoder** (vector/hybrid retrieval).
- **Tối ưu latency** bằng batching, truncation, warm-up và các “núm vặn” cấu hình.

## Baseline hiện tại (trong repo này)
- **Bi-encoder embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (xem `src/rag/embedding.py`).
- Luồng retrieval: `src/rag/pipeline.py::run_rag()` build retriever → `retriever.invoke(question)` → build context → gọi LLM.
- Chế độ: `vector` và `hybrid` (BM25 + FAISS, fusion bằng RRF trong `src/rag/retriever.py`).

## Kiến trúc mục tiêu
### Vì sao cần retrieval 2 tầng?
- **Bi-encoder retrieval** nhanh và scale tốt (recall tốt), nhưng ranking top-\(k\) có thể nhiễu.
- **Cross-encoder** đọc cặp `(query, chunk)` cùng lúc nên chấm relevance chuẩn hơn, nhưng tốn compute.

**Do đó**:
- Tầng A (**Recall**): retrieve một tập ứng viên rộng hơn \(k_\text{candidate}\) bằng retriever hiện có.
- Tầng B (**Precision**): cross-encoder rerank tập ứng viên và chọn top-\(k\) nhỏ để đưa vào LLM.

### Luồng đề xuất
1. Build retriever với `k = RETRIEVE_CANDIDATES`.
2. `candidates = retriever.invoke(question)`.
3. Nếu bật rerank: `docs = rerank(question, candidates)[:RERANK_TOP_K]`
   - nếu không: `docs = candidates[:FINAL_TOP_K]` (giữ behavior hiện tại).
4. Build context từ `docs` rồi gọi LLM.

## Các “núm vặn” cấu hình (Env Vars)
Thêm biến môi trường (qua `os.getenv`) để tune accuracy/latency mà không cần sửa code:
- **`RERANK_ENABLED`**: `true/false` (mặc định `false`)
- **`RERANK_MODEL`**: tên model cross-encoder (string)
- **`RETRIEVE_CANDIDATES`**: số candidate retrieve ban đầu trước rerank (mặc định ~30)
- **`RERANK_TOP_K`**: số chunk giữ lại sau rerank cho context LLM (mặc định ~4–8)
- **`RERANK_MAX_CHARS`**: cắt ngắn chunk trước khi scoring (mặc định ~1500–3000 chars)
- (tuỳ chọn) **`RERANK_BATCH_SIZE`**: batch size khi scoring (mặc định tuỳ device)
- (tuỳ chọn) **`RERANK_DEVICE`**: `cpu`/`cuda` nếu muốn ép device

## Các bước triển khai (chi tiết)
### Bước 1 — Thêm module reranker
Tạo `src/rag/reranker.py` gồm:
- Loader model kiểu **lazy/singleton** (không reload mỗi request).
- Hàm `rerank_documents(query, docs, top_k, max_chars, ...) -> (docs_sorted, scores?)`.
- **Batch scoring**:
  - Build `pairs = [(query, truncated_doc_text), ...]`
  - Gọi `CrossEncoder.predict(pairs, batch_size=...)`
  - Sort score giảm dần, trả về top-\(k\) docs.

Ghi chú:
- Truncation nên làm **trước** scoring để latency ổn định.
- Tách reranker khỏi pipeline để dễ test và dễ thay model.

### Bước 2 — Cắm rerank vào `run_rag`
Update `src/rag/pipeline.py::run_rag()`:
- Tách 2 giá trị \(k\):
  - `retrieve_k = RETRIEVE_CANDIDATES` (ví dụ 30)
  - `final_k = RERANK_TOP_K` (ví dụ 4)
- Build retriever với `k=retrieve_k` (cho cả vector và hybrid).
- Sau retrieval:
  - nếu bật: rerank rồi giữ `final_k`
  - nếu tắt: lấy trực tiếp `final_k` theo thứ tự retriever (baseline)

### Bước 3 — Đo latency theo từng stage
Trong `run_rag()`, đo riêng:
- `retrieval_ms`
- `rerank_ms` (0 nếu tắt)
- `llm_ms`
- `total_ms`

Expose theo 1 trong 2 cách:
- Option A: mở rộng model `SearchResult` với các field này (khuyến nghị để debug).
- Option B: chỉ log dạng structured logs (nhanh, nhưng API không thấy).

### Bước 4 — So sánh A/B công bằng (bi-encoder vs rerank)
Thêm helper so sánh (tương tự `compare_search_modes`) trả về:
- **Baseline**: cùng candidate pool, nhưng **không rerank** → top \(k\) theo retriever
- **Rerank**: cùng candidate pool → top \(k\) theo cross-encoder

Vì sao quan trọng:
- Nó cô lập đúng tác động của reranker lên ranking, không thay đổi retrieval stage.

### Bước 5 — Dependencies và runtime
Thêm/xác nhận dependency trong `requirements.txt`:
- `sentence-transformers` (để dùng `CrossEncoder`)
- `torch` (sentence-transformers cần)

Docker/runtime:
- CPU: dùng torch bản CPU.
- GPU (tuỳ chọn): torch CUDA + runtime phù hợp (để follow-up nếu deploy hiện tại là CPU).

## Checklist tối ưu latency (thực dụng)
Làm theo thứ tự:
1. **Batch scoring** (tránh forward từng doc).
2. **Truncate** theo `RERANK_MAX_CHARS`.
3. Tune `RETRIEVE_CANDIDATES` giảm dần cho tới khi chất lượng bắt đầu tụt.
4. **Warm-up** model lúc startup (tuỳ chọn nhưng khuyến nghị để tránh spike request đầu).
5. Nếu có GPU: chạy reranker trên GPU và giữ model loaded.

## Default đề xuất (bắt đầu nhanh rồi tune)
- `RERANK_ENABLED=false` (an toàn)
- `RETRIEVE_CANDIDATES=30`
- `RERANK_TOP_K=4` (giữ size context như hiện tại)
- `RERANK_MAX_CHARS=2000`
- Chọn cross-encoder nhỏ trước để validate flow/latency, rồi đổi model tốt hơn nếu cần.

## Definition of Done
- **Đúng chức năng**:
  - Bật `RERANK_ENABLED=true` thì thứ tự citations/context thay đổi theo rerank.
  - Tắt rerank thì behavior khớp baseline.
- **Có so sánh**:
  - Có output A/B (baseline vs rerank) cho cùng query và cùng candidate pool.
- **Có đo latency**:
  - Có số đo retrieval/rerank/LLM/total (trong API hoặc logs).
- **Ổn định hiệu năng**:
  - Không load model mỗi request (singleton).
  - Xử lý edge cases: candidate rỗng; `RERANK_TOP_K > len(candidates)`.

