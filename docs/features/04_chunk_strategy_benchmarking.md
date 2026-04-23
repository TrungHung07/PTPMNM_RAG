# [Feature 4] Tùy chỉnh Chunk Strategy & Benchmarking

**Ngày:** 2026-04-22

### Mục tiêu

Trước đây, hệ thống RAG sử dụng các tham số chunking cố định (`chunk_size=1000`, `overlap=200`). Tuy nhiên, mỗi loại tài liệu (ngắn, dài, chuyên sâu) lại đòi hỏi cách chia nhỏ khác nhau để đạt độ chính xác tối ưu.

Tính năng này cho phép người dùng tùy chỉnh tham số khi upload tài liệu, đồng thời cung cấp một hệ thống đánh giá (benchmarking) tự động để tìm ra cấu hình tốt nhất cho từng bộ dữ liệu.

### Kiến trúc thay đổi

```
  Người dùng/Frontend ─► POST /upload?chunk_size=N&chunk_overlap=M ─► build_index()
                                  │
                                  └─► Phân mảnh theo thông số mới
                                  └─► Lưu số lượng chunks tương ứng

  Hệ thống Evaluation:
    Tài liệu mẫu (chay_bo.pdf) ─► Loop qua 12 tổ hợp (Size 500-2000, Overlap 50-200)
                                 ─► Chạy câu hỏi bộ test (benchmark_data.json)
                                 ─► Tính Hit Rate based on Keyword Overlap
                                 ─► Xuất báo cáo benchmark_results.md
```

### Files thay đổi

| File | Loại | Mô tả |
|---|---|---|
| `app.py` | MODIFY | Cập nhật endpoint `/upload` nhận thêm `chunk_size` và `chunk_overlap` (Optional). |
| `tests/benchmark_data.json` | **NEW** | Bộ dữ liệu "Ground Truth" (câu hỏi + câu trả lời chuẩn) trích xuất từ `chay_bo.pdf`. |
| `benchmark_chunking.py` | **NEW** | Script tự động hóa việc thử nghiệm 12 tổ hợp tham số và đo lường độ chính xác (Hit Rate). |
| `benchmark_results.md` | **NEW** | Báo cáo chi tiết kết quả thử nghiệm cho từng cấu hình. |

### Kết quả Benchmark (Tóm tắt)

Thực hiện đánh giá trên tài liệu `chay_bo.pdf` với các tổ hợp:
- **Chunk sizes**: [500, 1000, 1500, 2000]
- **Overlaps**: [50, 100, 200]

**Phát hiện chính:**
- Hầu hết các cấu hình đều đạt **Hit Rate 1.0 (100%)** nhờ cơ chế Hybrid Search mạnh mẽ.
- Riêng cấu hình `chunk_size=500` với `overlap=50` chỉ đạt **0.75**, do mảnh quá nhỏ và không đủ thông tin bao quát để khớp từ khóa hiệu quả.
- `chunk_size=1000` với `overlap=200` vẫn tỏ ra là cấu hình cân bằng nhất giữa hiệu năng và độ chi tiết.

### Hướng dẫn sử dụng API

Người dùng có thể tinh chỉnh cấu hình ngay khi upload để tối ưu cho tài liệu đặc thù:

```bash
curl -X POST "http://localhost:8000/upload?chunk_size=500&chunk_overlap=100" -F "files=@document.pdf"
```

### Cách chạy Benchmark đánh giá

Để tự chạy lại các thử nghiệm trên các tài liệu khác hoặc bộ câu hỏi mới:

1.  **Cập nhật bộ câu hỏi**: Thêm các câu hỏi và câu trả lời kỳ vọng vào `tests/benchmark_data.json`.
2.  **Đảm bảo file PDF tồn tại**: Đặt file PDF cần test (mặc định là `chay_bo.pdf`) vào thư mục gốc.
3.  **Chạy script**:
    ```bash
    # Kích hoạt venv nếu cần
    .\venv\Scripts\activate
    
    # Chạy script benchmark
    python benchmark_chunking.py
    ```
4.  **Xem kết quả**: Báo cáo sẽ được sinh ra tại file `benchmark_results.md`.
