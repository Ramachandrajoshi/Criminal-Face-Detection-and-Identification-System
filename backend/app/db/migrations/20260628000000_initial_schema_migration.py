"""
Migration 001: Create all database tables and indexes.

Creates the pgvector extension, suspect_profiles, audit_log, and alerts tables,
plus the HNSW index for ANN cosine-distance search.
"""

from sqlalchemy import text


async def upgrade(session):
    # Enable pgvector extension (idempotent).
    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await session.commit()

    # suspect_profiles — stores encrypted 512-d face embeddings
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS suspect_profiles (
            id                 SERIAL PRIMARY KEY,
            person_name       VARCHAR(100) NOT NULL,
            alias              VARCHAR(100),
            demographics       JSONB,
            face_embedding     vector(512) NOT NULL,
            face_embedding_enc BYTEA,
            created_at         TIMESTAMPTZ DEFAULT now()
        )
    """))
    await session.commit()

    # HNSW index for approximate nearest-neighbour cosine search
    await session.execute(text("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'suspect_embedding_hnsw_idx'
            ) THEN
                CREATE INDEX suspect_embedding_hnsw_idx
                    ON suspect_profiles
                    USING hnsw (face_embedding vector_cosine_ops);
            END IF;
        END $$;
    """))
    await session.commit()

    # audit_log — append-only event log
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id           SERIAL PRIMARY KEY,
            event_type   VARCHAR(50) NOT NULL,
            query_hash   TEXT NOT NULL,
            result_name  VARCHAR(100),
            distance     FLOAT,
            gps_lat      FLOAT,
            gps_lon      FLOAT,
            timestamp    TIMESTAMPTZ DEFAULT now()
        )
    """))
    await session.commit()

    # alerts — human-in-the-loop confirmation queue
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS alerts (
            id            SERIAL PRIMARY KEY,
            audit_log_id  INTEGER REFERENCES audit_log(id),
            suspect_id    INTEGER,
            event_type    VARCHAR(50) NOT NULL,
            distance      FLOAT,
            status        VARCHAR(20) NOT NULL DEFAULT 'PENDING_REVIEW',
            gps_lat       FLOAT,
            gps_lon       FLOAT,
            created_at    TIMESTAMPTZ DEFAULT now(),
            confirmed_at  TIMESTAMPTZ
        )
    """))
    await session.commit()


async def downgrade(session):
    await session.execute(text("DROP TABLE IF EXISTS alerts"))
    await session.commit()
    await session.execute(text("DROP TABLE IF EXISTS audit_log"))
    await session.commit()
    await session.execute(text("DROP INDEX IF EXISTS suspect_embedding_hnsw_idx"))
    await session.commit()
    await session.execute(text("DROP TABLE IF EXISTS suspect_profiles"))
    await session.commit()
