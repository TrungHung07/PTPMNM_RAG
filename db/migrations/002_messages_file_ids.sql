-- ----------------------------
-- Migration 002: Thêm cột file_ids vào bảng messages
--
-- Mục đích:
--   Mỗi tin nhắn giờ ghi nhận tập file UUID nào đã được dùng để trả lời
--   ở lượt đó, cho phép Conversational RAG lọc history đúng ngữ cảnh.
--
-- Back-compat:
--   DEFAULT '{}'::uuid[] → các tin nhắn cũ không bị lỗi khi đọc.
--   Khi query, bỏ qua tin nhắn có file_ids = '{}' (history cũ vô ngữ cảnh).
--
-- Chạy một lần trên DB đang live:
--   psql -U raguser -d ragdb -f db/migrations/002_messages_file_ids.sql
-- ----------------------------

ALTER TABLE "public"."messages"
  ADD COLUMN IF NOT EXISTS "file_ids" uuid[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN "public"."messages"."file_ids" IS
  'Danh sách doc_id (uuid) đã dùng để trả lời lượt này; '
  'dùng để lọc conversational history theo từng tập file.';

-- GIN index để query overlap &&  nhanh trên mảng uuid
CREATE INDEX IF NOT EXISTS "idx_messages_file_ids"
  ON "public"."messages" USING GIN ("file_ids");
