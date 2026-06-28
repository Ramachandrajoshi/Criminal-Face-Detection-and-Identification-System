"""
Migration 004: Rename alerts.suspect_id to alerts.face_id.

Completes the suspect→face terminology rename by updating the foreign key
column in the alerts table to match the renamed face_profiles table.
"""

from sqlalchemy import text


async def upgrade(session):
    # Rename suspect_id to face_id in alerts table
    await session.execute(text("ALTER TABLE alerts RENAME COLUMN suspect_id TO face_id"))
    await session.commit()

    # Rename the FK constraint if it exists
    await session.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'alerts_suspect_id_fkey'
                  AND table_name = 'alerts'
            ) THEN
                ALTER TABLE alerts RENAME CONSTRAINT alerts_suspect_id_fkey TO alerts_face_id_fkey;
            END IF;
        END $$;
    """))
    await session.commit()


async def downgrade(session):
    # Rename face_id back to suspect_id
    await session.execute(text("ALTER TABLE alerts RENAME COLUMN face_id TO suspect_id"))
    await session.commit()

    # Rename the FK constraint back
    await session.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'alerts_face_id_fkey'
                  AND table_name = 'alerts'
            ) THEN
                ALTER TABLE alerts RENAME CONSTRAINT alerts_face_id_fkey TO alerts_suspect_id_fkey;
            END IF;
        END $$;
    """))
    await session.commit()
