"""
Migration: Drop the duplicate HNSW index on face_profiles.face_embedding.

`face_embedding_hnsw_tenant_idx` was added in the tenant-RLS migration
(20260628200000) as a nominally "partial" index with predicate
`WHERE (tenant_id IS NOT NULL)`. `tenant_id` is declared
`INTEGER NOT NULL DEFAULT 1` (same migration), so that predicate is always
true — the index is a full functional duplicate of the original
`face_embedding_hnsw_idx` (from 20260628000000, originally named
`suspect_embedding_hnsw_idx`). Both were maintained on every INSERT/UPDATE
into face_profiles for zero query benefit; `cosine_ann_query()` already
filters by tenant_id in the SQL WHERE clause regardless of which index
Postgres picks.

Keeps the original `face_embedding_hnsw_idx` to minimize churn.
"""

from sqlalchemy import text


async def upgrade(session):
    await session.execute(text("DROP INDEX IF EXISTS face_embedding_hnsw_tenant_idx"))
    await session.commit()


async def downgrade(session):
    await session.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'face_embedding_hnsw_tenant_idx'
            ) THEN
                CREATE INDEX face_embedding_hnsw_tenant_idx
                    ON face_profiles
                    USING hnsw (face_embedding vector_cosine_ops)
                    WHERE (tenant_id IS NOT NULL);
            END IF;
        END $$;
    """))
    await session.commit()
