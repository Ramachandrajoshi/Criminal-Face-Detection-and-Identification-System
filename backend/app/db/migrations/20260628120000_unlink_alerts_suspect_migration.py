"""
Migration 001: Unlink alerts from suspect_profiles.

Removes the FK constraint on ``alerts.suspect_id`` so that deleting a suspect
no longer fails when alerts reference them.  The ``suspect_id`` value on
existing alerts becomes nullable (the FK is dropped, the data is preserved).
"""

from sqlalchemy import text


async def upgrade(session):
    # Create the tracking table if it doesn't exist yet (safety net).
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id          SERIAL PRIMARY KEY,
            migration   VARCHAR(200) NOT NULL UNIQUE,
            applied_at  TIMESTAMPTZ DEFAULT now()
        )
    """))
    await session.commit()

    # Drop the FK constraint.  Uses IF EXISTS to be safe if already removed.
    await session.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'alerts_suspect_id_fkey'
                  AND table_name = 'alerts'
            ) THEN
                ALTER TABLE alerts DROP CONSTRAINT alerts_suspect_id_fkey;
            END IF;
        END $$;
    """))
    await session.commit()
