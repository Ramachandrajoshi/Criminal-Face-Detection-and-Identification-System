"""
Migration 002: Rename suspect_profiles to face_profiles.

Converts the domain-specific 'suspect' naming to generic 'face' terminology
for better flexibility and neutrality while maintaining criminal investigation context.
"""

from sqlalchemy import text


async def upgrade(session):
    # Rename the table from suspect_profiles to face_profiles
    await session.execute(text("ALTER TABLE suspect_profiles RENAME TO face_profiles"))
    await session.commit()

    # Rename the primary key sequence
    await session.execute(text("ALTER SEQUENCE suspect_profiles_id_seq RENAME TO face_profiles_id_seq"))
    await session.commit()

    # Rename the person_name column to face_name
    await session.execute(text("ALTER TABLE face_profiles RENAME COLUMN person_name TO face_name"))
    await session.commit()

    # Rename the HNSW index to match the new table name
    await session.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'suspect_embedding_hnsw_idx'
            ) THEN
                ALTER INDEX suspect_embedding_hnsw_idx RENAME TO face_embedding_hnsw_idx;
            END IF;
        END $$;
    """))
    await session.commit()


async def downgrade(session):
    # Rename the column back
    await session.execute(text("ALTER TABLE face_profiles RENAME COLUMN face_name TO person_name"))
    await session.commit()

    # Rename the table back
    await session.execute(text("ALTER TABLE face_profiles RENAME TO suspect_profiles"))
    await session.commit()

    # Rename the sequence back
    await session.execute(text("ALTER SEQUENCE face_profiles_id_seq RENAME TO suspect_profiles_id_seq"))
    await session.commit()

    # Rename the index back
    await session.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'face_embedding_hnsw_idx'
            ) THEN
                ALTER INDEX face_embedding_hnsw_idx RENAME TO suspect_embedding_hnsw_idx;
            END IF;
        END $$;
    """))
    await session.commit()
