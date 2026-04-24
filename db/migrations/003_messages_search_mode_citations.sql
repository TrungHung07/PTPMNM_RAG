-- ----------------------------
-- Migration 003: Thêm search_mode và citations vào bảng messages
--
-- Mục đích:
--   - search_mode: ghi nhận chiến lược retrieval đã dùng cho từng message
--     (vector / hybrid / compare_vector / compare_hybrid / unknown)
--   - citations: lưu danh sách nguồn trích dẫn mà AI dựa vào để trả lời,
--     shape mỗi phần tử: { content: str, metadata: {}, score?: float }
--
-- Back-compat:
--   - DEFAULT 'unknown' → các message cũ không bị lỗi khi đọc.
--   - DEFAULT '[]'::jsonb → citations rỗng cho message cũ.
--
-- Chạy một lần trên DB đang live:
--   Windows (PowerShell):
--     Get-Content db\migrations\003_messages_search_mode_citations.sql | docker exec -i ptpmnm_postgres psql -U raguser -d ragdb
--   macOS/Linux:
--     docker exec -i ptpmnm_postgres psql -U raguser -d ragdb < db/migrations/003_messages_search_mode_citations.sql
-- ----------------------------

ALTER TABLE "public"."messages"
  ADD COLUMN IF NOT EXISTS "search_mode" text NOT NULL DEFAULT 'unknown';

COMMENT ON COLUMN "public"."messages"."search_mode" IS
  'Chiến lược retrieval đã dùng: vector | hybrid | compare_vector | compare_hybrid | unknown';

ALTER TABLE "public"."messages"
  ADD COLUMN IF NOT EXISTS "citations" jsonb NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN "public"."messages"."citations" IS
  'Danh sách citation nguồn dùng để trả lời. '
  'Mỗi phần tử: { "content": "...", "metadata": {...}, "score": float | null }';

-- Index để filter / group theo search_mode (tuỳ chọn, hữu ích khi phân tích sau)
CREATE INDEX IF NOT EXISTS "idx_messages_search_mode"
  ON "public"."messages" ("search_mode");
