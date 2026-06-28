"""
SQLAlchemy ORM models.

All timestamp columns use ``DateTime(timezone=True)`` which maps to PostgreSQL
``TIMESTAMPTZ``.  This matches the migration schema and lets asyncpg accept
timezone-aware Python ``datetime`` objects without raising:
  "can't subtract offset-naive and offset-aware datetimes"
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Float, ForeignKey, Integer, LargeBinary, String, Text, text
from sqlalchemy import DateTime                      # re-imported cleanly below
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Timezone-aware UTC factory ───────────────────────────────────

def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


# ── Models ───────────────────────────────────────────────────────

class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id = Column(Integer, primary_key=True, index=True)
    face_name = Column(String(100), nullable=False)
    alias = Column(String(100), nullable=True)
    demographics = Column(JSONB, nullable=True)
    # Stored as pgvector vector(512) in DB; encrypted payload stored separately.
    face_embedding = Column(Text, nullable=False)
    face_embedding_enc = Column(LargeBinary, nullable=True)
    tenant_id = Column(Integer, nullable=False, server_default=text("1"))
    # TIMESTAMPTZ — must use timezone=True so asyncpg accepts tz-aware datetimes
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False)   # MATCH | NO_MATCH | REGISTER | SPOOF_BLOCKED
    query_hash = Column(Text, nullable=False)          # SHA-256 of raw embedding bytes
    result_name = Column(String(100), nullable=True)
    distance = Column(Float, nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    tenant_id = Column(Integer, nullable=False, server_default=text("1"))
    # TIMESTAMPTZ — must use timezone=True
    timestamp = Column(DateTime(timezone=True), default=_utcnow)

    alerts = relationship("Alert", back_populates="audit_log")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    audit_log_id = Column(Integer, ForeignKey("audit_log.id"), nullable=True)
    face_id = Column(Integer, nullable=True)
    event_type = Column(String(50), nullable=False)
    distance = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="PENDING_REVIEW")
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    tenant_id = Column(Integer, nullable=False, server_default=text("1"))
    # TIMESTAMPTZ — must use timezone=True
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)

    audit_log = relationship("AuditLog", back_populates="alerts")

