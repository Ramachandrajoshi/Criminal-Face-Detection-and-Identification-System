"""
Migration: index (tenant_id, embedding_version) on face_profiles.

Search embeds the query once per enrolled recipe, so before doing any of that
work the pipeline asks which ``embedding_version`` values actually have rows
for the tenant (``get_present_embedding_versions``). That census runs on every
search, so it must not become a sequential scan as the table grows.

``embedding_version`` is very low cardinality, so a composite index on
(tenant_id, embedding_version) turns the SELECT DISTINCT into an index-only
scan. No column or constraint changes — this is purely an access-path
improvement and is safe to apply to a live table.
"""

from sqlalchemy import text


async def upgrade(session):
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS face_profiles_tenant_version_idx
            ON face_profiles (tenant_id, embedding_version)
    """))
    await session.commit()


async def downgrade(session):
    await session.execute(text("""
        DROP INDEX IF EXISTS face_profiles_tenant_version_idx
    """))
    await session.commit()
