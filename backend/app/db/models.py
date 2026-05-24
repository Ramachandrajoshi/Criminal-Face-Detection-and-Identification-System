"""
SQLAlchemy ORM models.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class SuspectProfile(Base):
    __tablename__ = "suspect_profiles"

    id = Column(Integer, primary_key=True, index=True)
    suspect_name = Column(String(100), nullable=False)
    alias = Column(String(100), nullable=True)
    demographics = Column(JSONB, nullable=True)
    # Stored as pgvector vector(512) in DB; encrypted payload stored separately.
    face_embedding = Column(Text, nullable=False)
    face_embedding_enc = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    alerts = relationship("Alert", back_populates="suspect")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False)  # MATCH | NO_MATCH | REGISTER | SPOOF_BLOCKED
    query_hash = Column(Text, nullable=False)  # SHA-256 of raw embedding bytes
    result_name = Column(String(100), nullable=True)
    distance = Column(Float, nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # back-populate relationship (defined in models.py after class defs)
    alerts = relationship("Alert", back_populates="audit_log")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    audit_log_id = Column(Integer, ForeignKey("audit_log.id"), nullable=True)
    suspect_id = Column(Integer, ForeignKey("suspect_profiles.id"), nullable=True)
    event_type = Column(String(50), nullable=False)
    distance = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="PENDING_REVIEW")
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    confirmed_at = Column(DateTime, nullable=True)

    audit_log = relationship("AuditLog", back_populates="alerts")
    suspect = relationship("SuspectProfile", back_populates="alerts")
