# [Feature 3] Cross-Encoder Reranking & Citation Filtering

**Branch:** `chore/history`
**Ngày:** 2026-04-21

### Mục tiêu

Hệ thống RAG thông thường (hoặc ngay cả khi dùng Hybrid Search) có xu hướng trả về số lượng documents cố định (chẳng hạn `top_k = 4`). Điều này dẫn đến vấn đề **"rác citation" (Citation Noise)**: khi câu hỏi chỉ liên quan đến một lượng nhỏ chunks nhất định, hệ thống vẫn "vét cạn" cho đủ KPI 4 chunks từ các phần khác của tài liệu, gây nhiễu cho LLM và sai lệch thông tin trích dẫn trả về cho người dùng.

Tính năng này được ra đời để lọc bay các "rác citation" bằng cách sử dụng **Cross-Encoder Reranker**, một kiến trúc chấm điểm sự liên quan giữa Query và Chunk một cách độc lập và chính xác tuyệt đối.

### Kiến trúc thay đổi

```
  query ──► Retriever (Vector/Hybrid) ──► Top K ứng viên (docs)
                                                │
                                        Cross Encoder (Reranker)
                                        Tính tương quan Query - Chunk
                                                │
                 [chunk_1: 5.6], [chunk_2: 2.1], [chunk_3: -1.5], [chunk_4: -3.2]
                                                │
                                  Cắt ngưỡng (Score Threshold) >= 0.0
                                                │
                          Chỉ giữ lại [chunk_1, chunk_2] ──► LLM ──► Cực kỳ sạch sẽ
```

### Chi tiết thay đổi

- **`RERANK_ENABLED`:** Đã cấu hình và mặc định ép bật (True) trong code để hệ thống đi qua bước Rerank giúp lọc Citation. 
- **Lọc bằng Threshold (Cut-off):** Cập nhật `src/rag/rerank.py` bổ sung logic loại bỏ hoàn toàn (drop) các chunks có điểm số Cross-Encoder Logits thấp hơn ngưỡng `RERANK_SCORE_THRESHOLD` (ví dụ `0.0`), thay vì chỉ lấy Top K và giữ lại các chunks điểm thấp.
- Mô hình mặc định được sử dụng là `cross-encoder/ms-marco-MiniLM-L-6-v2` giúp cân bằng tốt giữa tốc độ xử lý (độ trễ khoảng 100-300ms) và hiệu năng chính xác cho tiếng Việt/Anh. Đảm bảo rằng API trả về không bao giờ bị "vét" thêm các files không liên quan đi kèm.
