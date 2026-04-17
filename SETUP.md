# SETUP GUIDE

Hướng dẫn thiết lập môi trường cho dự án RAG.

---

## Yêu cầu

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) đã cài và đang chạy
- Python 3.11+
- Git

---

## Cách 1: Chạy toàn bộ bằng Docker (khuyến nghị)

### Bước 1 — Clone và chuẩn bị env

```bash
git clone <repo-url>
cd PTPMNM_RAG

cp .env.example .env
# Chỉnh sửa .env nếu cần thay đổi password/port
```

### Bước 2 — Khởi động tất cả services

```bash
docker compose up -d
```

Lệnh này sẽ tự động:
- Khởi động **PostgreSQL** và chạy migration tạo bảng (`db/migrations/001_init.sql`)
- Khởi động **Ollama**
- Build và khởi động **FastAPI app**

### Bước 3 — Pull model AI (chỉ cần làm 1 lần)

```bash
docker exec -it ptpmnm_ollama ollama pull qwen2.5:3b
```

> Lần đầu sẽ tải model ~2GB. Các lần sau model đã được cache trong volume `ollama_data`.

### Bước 4 — Kiểm tra

```bash
# Xem trạng thái các container
docker compose ps

# Xem log
docker compose logs -f app
```

Mở trình duyệt: **http://localhost:8000/docs**

---

## Cách 2: Chạy app local, DB + Ollama trong Docker

Phù hợp cho việc phát triển (hot-reload, debug dễ hơn).

### Bước 1 — Khởi động DB và Ollama

```bash
docker compose up postgres ollama -d
```

### Bước 2 — Pull model AI (chỉ cần làm 1 lần)

```bash
docker exec -it ptpmnm_ollama ollama pull qwen2.5:3b
```

### Bước 3 — Cài dependencies Python

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Bước 4 — Cấu hình env và chạy app

```bash
cp .env.example .env
# Đảm bảo DATABASE_URL và OLLAMA_BASE_URL dùng localhost (đã là mặc định)

uvicorn app:app --reload
```

Mở trình duyệt: **http://localhost:8000/docs**

---

## Seed dữ liệu mẫu (optional)

Sau khi containers đã chạy, chạy lệnh sau để tạo dữ liệu mẫu cho development:

```bash
# Windows (PowerShell)
Get-Content db\seeds\001_seed.sql | docker exec -i ptpmnm_postgres psql -U raguser -d ragdb

# macOS/Linux
docker exec -i ptpmnm_postgres psql -U raguser -d ragdb < db/seeds/001_seed.sql
```

---

## Các lệnh hữu ích

```bash
# Dừng tất cả services
docker compose down

# Dừng và xóa toàn bộ data (reset hoàn toàn)
docker compose down -v

# Xem log từng service
docker compose logs postgres
docker compose logs ollama
docker compose logs app

# Kết nối trực tiếp vào PostgreSQL
docker exec -it ptpmnm_postgres psql -U raguser -d ragdb

# Xem danh sách model Ollama đã tải
docker exec -it ptpmnm_ollama ollama list

# Rebuild app sau khi thay đổi code
docker compose build app
docker compose up app -d
```

---

## Cấu trúc thư mục

```
PTPMNM_RAG/
├── app.py                    # FastAPI entry point
├── docker-compose.yml        # Định nghĩa services
├── Dockerfile                # Build FastAPI app
├── requirements.txt
├── .env.example              # Template biến môi trường
├── SETUP.md                  # File này
├── db/
│   ├── migrations/
│   │   └── 001_init.sql      # Tạo bảng (tự chạy khi postgres khởi động)
│   └── seeds/
│       └── 001_seed.sql      # Dữ liệu mẫu (chạy thủ công)
└── src/
    ├── database.py           # PostgreSQL connection pool & queries
    ├── history/
    │   └── router.py         # Chat history APIs
    └── rag/
        ├── pipeline.py       # Conversational RAG logic
        └── llm.py            # Ollama LLM client
```

---

## Troubleshooting

**Lỗi: `connection refused` khi app kết nối postgres**
- Đảm bảo postgres container đang healthy: `docker compose ps`
- Kiểm tra `DATABASE_URL` trong `.env` đúng host (`localhost` nếu chạy local, `postgres` nếu trong docker)

**Lỗi: Ollama model not found**
- Chạy lại: `docker exec -it ptpmnm_ollama ollama pull qwen2.5:3b`

**Lỗi: Port 5432 đã bị chiếm**
- Thay đổi port mapping trong `docker-compose.yml`: `"5433:5432"` và cập nhật `DATABASE_URL` tương ứng
