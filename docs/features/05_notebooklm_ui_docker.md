# [Feature 5] Giao diện NotebookLM & Dockerization

**Ngày:** 2026-04-22

### Mục tiêu

Chuyển đổi dự án từ một công cụ CLI thành một ứng dụng web hoàn chỉnh với giao diện người dùng trực quan và khả năng triển khai "một chạm" qua Docker.

Tính năng này tập trung vào việc cải thiện trải nghiệm người dùng (UX) và đơn giản hóa quy trình cài đặt hệ thống RAG phức tạp.

### Các thay đổi chính

#### 1. Giao diện người dùng (Streamlit Frontend)
- **Bố cục 2 cột**: Mô phỏng NotebookLM với "Source Explorer" bên trái và "Chat Panel" bên phải.
- **Evaluation Mode**: Cho phép so sánh song song Vector vs Hybrid Search trong cùng một màn hình.
- **Session Restore**: Tự động khôi phục lịch sử chat và danh sách tài liệu từ cơ sở dữ liệu khi vào lại phiên cũ.
- **Citation Chips**: Hiển thị nguồn trích dẫn dưới dạng các thẻ màu sắc, rê chuột để xem nhanh nội dung.

#### 2. Docker Orchestration
- **Full Stack Docker**: Hợp nhất Backend (FastAPI), Frontend (Streamlit), Database (Postgres) và AI Engine (Ollama).
- **Auto-pull Model**: Tự động tải mô hình AI (`qwen2.5`) ngay khi khởi động container lần đầu.
- **Dữ liệu bền vững (Persistence)**: Cấu hình volumes để không mất dữ liệu chat và index khi restart container.

### Files thay đổi

| File | Loại | Mô tả |
|---|---|---|
| `app_fe.py` | **NEW** | Giao diện Streamlit chính. |
| `Dockerfile` | MODIFY | Tối ưu hóa để chạy cả Backend và Frontend. |
| `docker-compose.yml` | MODIFY | Cấu hình 4 services chính + helper auto-pull. |
| `src/models.py` | MODIFY | Thêm `latency_ms` vào `AskResponse`. |
| `src/rag/pipeline.py` | MODIFY | Tính toán và trả về latency trong hàm `ask_question`. |
| `README.md` | MODIFY | Hướng dẫn sử dụng Docker mới. |

### Hướng dẫn chạy (Docker)

```bash
docker compose up --build
```
- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
