"""
Migration: Add embedding_version column to face_profiles table.

This versioning tracks which preprocessing pipeline was used to extract
the embedding. Version 1 is the old default pipeline; Version 2 includes
the enhanced preprocessing (upscale, denoise, gamma, unsharp).
"""

from sqlalchemy import text


async def upgrade(session):
    # Add embedding_version with a default of 1 for all existing rows
    await session.execute(text("""
        ALTER TABLE face_profiles 
        ADD COLUMN IF NOT EXISTS embedding_version INTEGER NOT NULL DEFAULT 1
    """))
    await session.commit()


async def downgrade(session):
    await session.execute(text("""
        ALTER TABLE face_profiles 
        DROP COLUMN IF EXISTS embedding_version
    """))
    await session.commit()
