-- Seed 001: Sample data for development/testing
-- Chạy thủ công: docker exec -i ptpmnm_postgres psql -U raguser -d ragdb < db/seeds/001_seed.sql

INSERT INTO sessions (file_id, filename, created_at) VALUES
    ('00000000-0000-0000-0000-000000000001', 'sample.docx',    NOW() - INTERVAL '2 days'),
    ('00000000-0000-0000-0000-000000000002', 'sample2.pdf',   NOW() - INTERVAL '1 day')
ON CONFLICT (file_id) DO NOTHING;

INSERT INTO messages (file_id, question, answer, created_at) VALUES
    ('00000000-0000-0000-0000-000000000001', 'Tài liệu này nói về gì?',       'Tài liệu mô tả...', NOW() - INTERVAL '2 days'),
    ('00000000-0000-0000-0000-000000000001', 'Có bao nhiêu chương?',           'Tài liệu có 3 chương...', NOW() - INTERVAL '2 days'),
    ('00000000-0000-0000-0000-000000000002', 'Tóm tắt nội dung chính?',        'Nội dung chính là...', NOW() - INTERVAL '1 day')
ON CONFLICT DO NOTHING;
