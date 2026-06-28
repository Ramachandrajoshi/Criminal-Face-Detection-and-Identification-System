"""
Lightweight migration runner.

Scans ``backend/app/db/migrations/`` for ``*_migration.py`` files,
executes pending ones in sorted (timestamp) order on server startup,
and tracks which ones have already been applied via the
``_migrations`` table (created on first run).

How to add a migration:
    1. Create ``backend/app/db/migrations/<YYYYMMDDHHMMSS>_slug.py``
    2. Implement ``async def upgrade(session)`` and optionally ``async def downgrade(session)``
    3. Restart the server — pending migrations run automatically.

The runner uses a raw SQL connection (via ``asyncpg``) for DDL operations
that SQLAlchemy's async engine may not handle gracefully (e.g. ``DROP CONSTRAINT``).
"""

import importlib.util
import logging
import re
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent
MIGRATIONS_TABLE = "_migrations"

# Regex: <timestamp>_<slug>.py
MIGRATION_PATTERN = re.compile(
    r"^(\d{14})_(.+?)_migration\.py$"
)


# ── Tracking table ──────────────────────────────────────────────


async def _ensure_tracking_table(session: AsyncSession) -> None:
    """Create the migrations tracking table if it doesn't exist."""
    await session.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
            id          SERIAL PRIMARY KEY,
            migration   VARCHAR(200) NOT NULL UNIQUE,
            applied_at  TIMESTAMPTZ DEFAULT now()
        )
    """))
    await session.commit()


# ── Discover migrations ────────────────────────────────────────


def _discover_migrations() -> list[tuple[str, str, Path]]:
    """
    Return sorted list of ``(timestamp, slug, path)`` for every migration
    file in ``MIGRATIONS_DIR``.
    """
    migrations: list[tuple[str, str, Path]] = []
    for f in sorted(MIGRATIONS_DIR.glob("*_migration.py")):
        match = MIGRATION_PATTERN.match(f.name)
        if match:
            migrations.append((match.group(1), match.group(2), f))
        else:
            logger.warning(
                "Skipping file %s — does not match naming pattern "
                "'<YYYYMMDDHHMMSS>_<slug>_migration.py'", f.name
            )
    return migrations


# ── Load & run ─────────────────────────────────────────────────


async def _load_migration(path: Path):
    """Dynamically import a migration module from disk."""
    spec = importlib.util.spec_from_file_location(
        f"migration_{path.stem}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _is_applied(session: AsyncSession, migration: str) -> bool:
    """Check whether a migration has already been applied."""
    result = await session.execute(
        text(f"SELECT 1 FROM {MIGRATIONS_TABLE} WHERE migration = :m"),
        {"m": migration},
    )
    return result.fetchone() is not None


async def _mark_applied(session: AsyncSession, migration: str) -> None:
    """Record that a migration has been applied."""
    await session.execute(
        text(f"""
            INSERT INTO {MIGRATIONS_TABLE} (migration)
            VALUES (:m)
            ON CONFLICT (migration) DO NOTHING
        """),
        {"m": migration},
    )
    await session.commit()


async def run_migrations(session: AsyncSession) -> dict:
    """
    Run all pending migrations in order.

    Returns a summary dict with counts and details.
    """
    await _ensure_tracking_table(session)

    discovered = _discover_migrations()
    if not discovered:
        logger.info("No migration files found in %s", MIGRATIONS_DIR)
        return {"applied": 0, "skipped": 0, "errors": []}

    applied = 0
    skipped = 0
    errors: list[dict] = []

    for timestamp, slug, path in discovered:
        migration_id = f"{timestamp}_{slug}"

        if await _is_applied(session, migration_id):
            logger.info("Skipping already-applied migration: %s", migration_id)
            skipped += 1
            continue

        logger.info("Running migration: %s", migration_id)

        try:
            mod = await _load_migration(path)
            if not hasattr(mod, "upgrade"):
                raise ValueError(f"Migration {migration_id} has no 'upgrade' function")

            await mod.upgrade(session)
            await _mark_applied(session, migration_id)
            applied += 1
            logger.info("Migration applied: %s", migration_id)

        except Exception as exc:
            logger.error("Migration failed: %s — %s", migration_id, exc)
            errors.append({"migration": migration_id, "error": str(exc)})
            break  # stop on first error

    return {
        "applied": applied,
        "skipped": skipped,
        "errors": errors,
    }
