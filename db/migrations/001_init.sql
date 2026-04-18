/*
 Navicat Premium Data Transfer

 Source Server         : Rag
 Source Server Type    : PostgreSQL
 Source Server Version : 160007 (160007)
 Source Host           : localhost:5432
 Source Catalog        : ragdb
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 160007 (160007)
 File Encoding         : 65001

 Date: 18/04/2026 11:37:59
*/

-- ----------------------------
-- Sequence structure for messages_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."messages_id_seq";
CREATE SEQUENCE "public"."messages_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 2147483647
START 1
CACHE 1;

-- ----------------------------
-- Table structure for documents
-- ----------------------------
DROP TABLE IF EXISTS "public"."documents";
CREATE TABLE "public"."documents" (
  "doc_id" uuid NOT NULL,
  "session_id" uuid NOT NULL,
  "file_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "uploaded_at" timestamptz(6) NOT NULL DEFAULT now(),
  "file_type" text COLLATE "pg_catalog"."default" NOT NULL
)
;
COMMENT ON COLUMN "public"."documents"."session_id" IS 'FK tới phiên; cho phép nhiều doc_id trùng session_id.';
COMMENT ON TABLE "public"."documents" IS 'Một dòng = một file đã upload; cùng session_id = cùng phiên (multi-file).';

-- ----------------------------
-- Table structure for messages
-- ----------------------------
DROP TABLE IF EXISTS "public"."messages";
CREATE TABLE "public"."messages" (
  "id" int4 NOT NULL DEFAULT nextval('messages_id_seq'::regclass),
  "session_id" uuid NOT NULL,
  "question" text COLLATE "pg_catalog"."default" NOT NULL,
  "answer" text COLLATE "pg_catalog"."default" NOT NULL,
  "created_at" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for sessions
-- ----------------------------
DROP TABLE IF EXISTS "public"."sessions";
CREATE TABLE "public"."sessions" (
  "session_id" uuid NOT NULL,
  "created_at" timestamptz(6) NOT NULL DEFAULT now()
)
;
COMMENT ON TABLE "public"."sessions" IS 'Một dòng = một phiên chat; không phải một file.';

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
ALTER SEQUENCE "public"."messages_id_seq"
OWNED BY "public"."messages"."id";
SELECT setval('"public"."messages_id_seq"', 1, false);

-- ----------------------------
-- Indexes structure for table documents
-- ----------------------------
CREATE INDEX "idx_documents_file_type" ON "public"."documents" USING btree (
  "file_type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_documents_session_id" ON "public"."documents" USING btree (
  "session_id" "pg_catalog"."uuid_ops" ASC NULLS LAST
);
CREATE INDEX "idx_documents_uploaded_at" ON "public"."documents" USING btree (
  "uploaded_at" "pg_catalog"."timestamptz_ops" DESC NULLS FIRST
);

-- ----------------------------
-- Primary Key structure for table documents
-- ----------------------------
ALTER TABLE "public"."documents" ADD CONSTRAINT "documents_pkey" PRIMARY KEY ("doc_id");

-- ----------------------------
-- Indexes structure for table messages
-- ----------------------------
CREATE INDEX "idx_messages_created_at" ON "public"."messages" USING btree (
  "created_at" "pg_catalog"."timestamptz_ops" ASC NULLS LAST
);
CREATE INDEX "idx_messages_session_id" ON "public"."messages" USING btree (
  "session_id" "pg_catalog"."uuid_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table messages
-- ----------------------------
ALTER TABLE "public"."messages" ADD CONSTRAINT "messages_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table sessions
-- ----------------------------
ALTER TABLE "public"."sessions" ADD CONSTRAINT "sessions_pkey" PRIMARY KEY ("session_id");

-- ----------------------------
-- Foreign Keys structure for table documents
-- ----------------------------
ALTER TABLE "public"."documents" ADD CONSTRAINT "documents_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "public"."sessions" ("session_id") ON DELETE CASCADE ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table messages
-- ----------------------------
ALTER TABLE "public"."messages" ADD CONSTRAINT "messages_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "public"."sessions" ("session_id") ON DELETE CASCADE ON UPDATE NO ACTION;
