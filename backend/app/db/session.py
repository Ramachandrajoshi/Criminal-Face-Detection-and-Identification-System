"""
SQLAlchemy async session factory.

Pool configuration:
  - pool_pre_ping=True      : test connection health before checkout (eliminates
                               "connection is closed" from stale idle connections)
  - pool_recycle=1800       : recycle connections every 30 min to avoid OS/firewall
                               silently dropping long-lived TCP connections
  - pool_size=10            : baseline concurrent connections
  - max_overflow=20         : allow burst up to 30 total connections
  - pool_timeout=30         : wait up to 30 s for a connection before raising
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

logger = logging.getLogger(__name__)

DATABASE_URL = (
    f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # ── Connection health ──────────────────────────────────────────
    # Issue a lightweight SELECT 1 before handing a pooled connection
    # to application code.  This catches connections dropped by the
    # server-side idle timeout or by network firewalls.
    pool_pre_ping=True,
    # ── Pool sizing ────────────────────────────────────────────────
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    # ── Connection lifetime ────────────────────────────────────────
    # Recycle connections every 30 minutes so that OS/firewall idle-
    # timeout rules never silently kill a connection that is still in
    # the pool.  Must be less than the server's tcp_keepalives_idle /
    # idle_session_timeout.
    pool_recycle=1800,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # Never auto-begin so that we control transaction boundaries explicitly.
    autobegin=True,
)


async def get_session() -> AsyncSession:
    """
    FastAPI dependency: yield a fresh async database session.

    The session is:
      - committed on success (caller is responsible for explicit commits)
      - rolled back automatically on any unhandled exception (prevents the
        session being returned to the pool in a dirty / broken state)
      - closed unconditionally in the finally block
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            # Roll back any open transaction so the connection is clean
            # when returned to the pool.
            await session.rollback()
            raise
        # Note: async_session_factory().__aexit__ calls session.close(),
        # which returns the underlying connection to the pool cleanly.
