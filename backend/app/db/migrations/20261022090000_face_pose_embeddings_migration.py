"""
Migration: face_pose_embeddings — several pose-tagged embeddings per profile.

Backs the pose-aware endpoints (/api/v1/pose/*). Purely additive: a new table
with a foreign key to face_profiles; no existing column, index or policy is
touched, so the current register / search flows are unaffected.

ON DELETE CASCADE keeps the existing DELETE /faces/{id} endpoint correct — a
deleted profile takes its pose embeddings with it.
"""

from sqlalchemy import text


async def upgrade(session):
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS face_pose_embeddings (
            id                 SERIAL PRIMARY KEY,
            face_id            INTEGER NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
            tenant_id          INTEGER NOT NULL DEFAULT 1,
            embedding          vector(512) NOT NULL,
            embedding_enc      BYTEA,
            embedding_version  INTEGER NOT NULL DEFAULT 4,
            pose_bucket        VARCHAR(32) NOT NULL,
            yaw                REAL,
            pitch              REAL,
            roll               REAL,
            quality            REAL,
            created_at         TIMESTAMPTZ DEFAULT now()
        )
    """))
    await session.commit()

    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS face_pose_embeddings_face_idx
            ON face_pose_embeddings (tenant_id, face_id)
    """))
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS face_pose_embeddings_hnsw_idx
            ON face_pose_embeddings
            USING hnsw (embedding vector_cosine_ops)
    """))
    await session.commit()

    # Same tenant isolation as face_profiles (see tenant_id_rls migration).
    await session.execute(text("ALTER TABLE face_pose_embeddings ENABLE ROW LEVEL SECURITY"))
    await session.execute(text(
        "DROP POLICY IF EXISTS face_pose_embeddings_tenant_access ON face_pose_embeddings"
    ))
    await session.execute(text("""
        CREATE POLICY face_pose_embeddings_tenant_access ON face_pose_embeddings
        USING (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
        WITH CHECK (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
    """))
    await session.commit()


async def downgrade(session):
    await session.execute(text("DROP TABLE IF EXISTS face_pose_embeddings"))
    await session.commit()
