-- Criminal Face Detection & Identification System
-- Database Schema
-- DO NOT modify without a migration

CREATE EXTENSION IF NOT EXISTS vector;

-- Suspect profiles table: stores encrypted 512-d face embeddings
CREATE TABLE suspect_profiles (
    id                 SERIAL PRIMARY KEY,
    suspect_name       VARCHAR(100) NOT NULL,
    alias              VARCHAR(100),
    demographics       JSONB,
    face_embedding     vector(512) NOT NULL,
    face_embedding_enc BYTEA,
    created_at         TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast approximate nearest neighbour search
CREATE INDEX suspect_embedding_hnsw_idx
    ON suspect_profiles
    USING hnsw (face_embedding vector_cosine_ops);

-- Immutable audit log (INSERT only — no UPDATE or DELETE)
CREATE TABLE audit_log (
    id           SERIAL PRIMARY KEY,
    event_type   VARCHAR(50) NOT NULL,    -- 'MATCH' | 'NO_MATCH' | 'REGISTER' | 'SPOOF_BLOCKED'
    query_hash   TEXT NOT NULL,           -- SHA-256 of raw embedding bytes
    result_name  VARCHAR(100),
    distance     FLOAT,
    gps_lat      FLOAT,
    gps_lon      FLOAT,
    timestamp    TIMESTAMPTZ DEFAULT now()
);

-- Alerts table for human-in-the-loop confirmation
CREATE TABLE alerts (
    id            SERIAL PRIMARY KEY,
    audit_log_id  INTEGER REFERENCES audit_log(id),
    suspect_id    INTEGER REFERENCES suspect_profiles(id),
    event_type    VARCHAR(50) NOT NULL,
    distance      FLOAT,
    status        VARCHAR(20) NOT NULL DEFAULT 'PENDING_REVIEW',
    gps_lat       FLOAT,
    gps_lon       FLOAT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    confirmed_at  TIMESTAMPTZ
);

-- Grant rules (enforce append-only on audit_log)
-- These are applied via pg_roles in production; documented here for clarity.
-- audit_log: SELECT + INSERT only
-- suspect_profiles: SELECT + INSERT
-- alerts: SELECT + UPDATE (status transition)
