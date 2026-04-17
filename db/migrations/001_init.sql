-- Migration 001: Initial schema
-- Chạy tự động khi postgres container khởi động lần đầu

CREATE TABLE IF NOT EXISTS sessions (
    file_id     UUID PRIMARY KEY,
    filename    TEXT NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id          SERIAL PRIMARY KEY,
    file_id     UUID NOT NULL REFERENCES sessions(file_id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_file_id ON messages(file_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
