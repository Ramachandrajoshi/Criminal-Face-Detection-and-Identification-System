"""
Migration 003: Add tenant_id with Row Level Security.

Adds a tenant_id column (default 1) to face_profiles, alerts, and audit_log.
Enables Row Level Security on all three tables so queries are automatically
filtered to the requesting user's tenant.

The tenant_id is read from the JWT token's "tenant" claim (set by the auth
middleware).  If the claim is absent the database-level default of 1 is used.
"""

from sqlalchemy import text


async def upgrade(session):
    # ── Ensure tracking table exists ──────────────────────────────
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id          SERIAL PRIMARY KEY,
            migration   VARCHAR(200) NOT NULL UNIQUE,
            applied_at  TIMESTAMPTZ DEFAULT now()
        )
    """))
    await session.commit()

    # ── face_profiles: add tenant_id column ───────────────────────
    await session.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'face_profiles' AND column_name = 'tenant_id'
            ) THEN
                ALTER TABLE face_profiles ADD COLUMN tenant_id INTEGER NOT NULL DEFAULT 1;
            END IF;
        END $$;
    """))
    await session.commit()

    # ── alerts: add tenant_id column ──────────────────────────────
    await session.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'alerts' AND column_name = 'tenant_id'
            ) THEN
                ALTER TABLE alerts ADD COLUMN tenant_id INTEGER NOT NULL DEFAULT 1;
            END IF;
        END $$;
    """))
    await session.commit()

    # ── audit_log: add tenant_id column ───────────────────────────
    await session.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'audit_log' AND column_name = 'tenant_id'
            ) THEN
                ALTER TABLE audit_log ADD COLUMN tenant_id INTEGER NOT NULL DEFAULT 1;
            END IF;
        END $$;
    """))
    await session.commit()

    # ── HNSW index on (face_embedding, tenant_id) for filtered ANN ─
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

    # ── Enable Row Level Security ─────────────────────────────────
    await session.execute(text("ALTER TABLE face_profiles ENABLE ROW LEVEL SECURITY"))
    await session.commit()

    await session.execute(text("ALTER TABLE alerts ENABLE ROW LEVEL SECURITY"))
    await session.commit()

    await session.execute(text("ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY"))
    await session.commit()

    # ── Policies: read/write scoped to current tenant ─────────────
    # pg_user_name() is not available; we rely on the JWT tenant claim
    # which the app injects via SET LOCAL jwt.claims.tenant = N.
    # The policies reference current_setting('jwt.claims.tenant', true)
    # which the middleware sets per-request.

    # face_profiles policies
    await session.execute(text("DROP POLICY IF EXISTS face_profiles_tenant_access ON face_profiles"))
    await session.execute(text("""
        CREATE POLICY face_profiles_tenant_access ON face_profiles
        USING (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
        WITH CHECK (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
    """))
    await session.commit()

    # alerts policies
    await session.execute(text("DROP POLICY IF EXISTS alerts_tenant_access ON alerts"))
    await session.execute(text("""
        CREATE POLICY alerts_tenant_access ON alerts
        USING (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
        WITH CHECK (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
    """))
    await session.commit()

    # audit_log policies (still append-only — only INSERT/SELECT allowed)
    await session.execute(text("DROP POLICY IF EXISTS audit_log_tenant_access ON audit_log"))
    await session.execute(text("""
        CREATE POLICY audit_log_tenant_select ON audit_log
        FOR SELECT
        USING (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
    """))
    await session.execute(text("""
        CREATE POLICY audit_log_tenant_insert ON audit_log
        FOR INSERT
        WITH CHECK (tenant_id = current_setting('jwt.claims.tenant', true)::INTEGER)
    """))
    await session.commit()


async def downgrade(session):
    # Revoke policies
    await session.execute(text("DROP POLICY IF EXISTS face_profiles_tenant_access ON face_profiles"))
    await session.commit()
    await session.execute(text("DROP POLICY IF EXISTS alerts_tenant_access ON alerts"))
    await session.commit()
    await session.execute(text("DROP POLICY IF EXISTS audit_log_tenant_access ON audit_log"))
    await session.execute(text("DROP POLICY IF EXISTS audit_log_tenant_select ON audit_log"))
    await session.execute(text("DROP POLICY IF EXISTS audit_log_tenant_insert ON audit_log"))
    await session.commit()

    # Disable RLS
    await session.execute(text("ALTER TABLE audit_log DISABLE ROW LEVEL SECURITY"))
    await session.commit()
    await session.execute(text("ALTER TABLE alerts DISABLE ROW LEVEL SECURITY"))
    await session.commit()
    await session.execute(text("ALTER TABLE face_profiles DISABLE ROW LEVEL SECURITY"))
    await session.commit()

    # Drop HNSW index with tenant filter
    await session.execute(text("DROP INDEX IF EXISTS face_embedding_hnsw_tenant_idx"))
    await session.commit()

    # Remove tenant_id columns
    await session.execute(text("ALTER TABLE audit_log DROP COLUMN IF EXISTS tenant_id"))
    await session.commit()
    await session.execute(text("ALTER TABLE alerts DROP COLUMN IF EXISTS tenant_id"))
    await session.commit()
    await session.execute(text("ALTER TABLE face_profiles DROP COLUMN IF EXISTS tenant_id"))
    await session.commit()
