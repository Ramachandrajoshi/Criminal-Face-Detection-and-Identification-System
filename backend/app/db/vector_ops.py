"""
pgvector ANN query helpers.
Handles AES-256 decryption of stored embeddings before cosine similarity search.
"""

import hashlib
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import text

from app.core.config import settings
from app.core.crypto import decrypt_embedding_bytes, encrypt_embedding_vector


def compute_query_hash(embedding_bytes: bytes) -> str:
    """SHA-256 hash of raw embedding byte array."""
    return hashlib.sha256(embedding_bytes).hexdigest()


async def cosine_ann_query(
    session,
    query_vector: list[float] | bytes,
    threshold: Optional[float] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Perform approximate nearest-neighbour cosine similarity search
    using the pgvector HNSW index.

    Uses the pgvector `<=>` operator which returns COSINE DISTANCE
    (0 = identical, 2 = opposite). Lower distance = better match.

    Returns a list of dicts: {id, suspect_name, alias, distance, ...}
    sorted by ascending distance (closest match first).

    threshold defaults to MATCH_THRESHOLD from config.
    """
    threshold = threshold if threshold is not None else settings.match_threshold

    if isinstance(query_vector, (bytes, bytearray)):
        query_vector = decrypt_embedding_bytes(bytes(query_vector)).tolist()

    # Build vector literal from list for pgvector
    vec_str = f"[{','.join(f'{v:.10f}' for v in query_vector)}]"

    # Cosine distance via <=> operator (lower is better match)
    # Filter: distance <= threshold (e.g., 0.58)
    sql = text(
        """
        SELECT
            id,
            suspect_name,
            alias,
            (face_embedding <=> :vec) AS distance
        FROM suspect_profiles
        WHERE (face_embedding <=> :vec) <= :threshold
        ORDER BY distance ASC
        LIMIT :limit
        """
    )

    result = await session.execute(
        sql,
        {"vec": vec_str, "threshold": threshold, "limit": limit},
    )

    rows = []
    for row in result:
        rows.append({
            "id": row[0],
            "suspect_name": row[1],
            "alias": row[2],
            "distance": round(float(row[3]), 6),
        })

    return rows


async def register_profile(
    session,
    suspect_name: str,
    alias: Optional[str],
    demographics: Optional[dict],
    embedding: np.ndarray | list[float],
) -> int:
    """
    Insert a new suspect profile.
    Embeddings are stored as pgvector plus AES-256 encrypted payload at rest.
    """
    vector = np.asarray(embedding, dtype=np.float32)
    vec_str = f"[{','.join(f'{v:.10f}' for v in vector.tolist())}]"
    encrypted = encrypt_embedding_vector(vector)

    sql = text(
        """
        INSERT INTO suspect_profiles (suspect_name, alias, demographics, face_embedding, face_embedding_enc)
        VALUES (:suspect_name, :alias, :demographics, CAST(:vec AS vector), :enc)
        RETURNING id
        """
    )

    result = await session.execute(
        sql,
        {
            "suspect_name": suspect_name,
            "alias": alias,
            "demographics": demographics,
            "vec": vec_str,
            "enc": encrypted,
        },
    )

    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return int(result.scalar_one())


async def add_audit_entry(
    session,
    event_type: str,
    query_hash: str,
    result_name: Optional[str] = None,
    distance: Optional[float] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
) -> int:
    """Append-only: insert an audit log entry."""
    from app.db.models import AuditLog

    entry = AuditLog(
        event_type=event_type,
        query_hash=query_hash,
        result_name=result_name,
        distance=distance,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )
    session.add(entry)
    try:
        await session.commit()
        await session.refresh(entry)
    except Exception:
        await session.rollback()
        raise
    return entry.id


async def create_alert(
    session,
    audit_log_id: int,
    suspect_id: Optional[int],
    event_type: str,
    distance: Optional[float],
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
) -> int:
    """Create an alert for human-in-the-loop review."""
    from app.db.models import Alert

    alert = Alert(
        audit_log_id=audit_log_id,
        suspect_id=suspect_id,
        event_type=event_type,
        distance=distance,
        status="PENDING_REVIEW",
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )
    session.add(alert)
    try:
        await session.commit()
        await session.refresh(alert)
    except Exception:
        await session.rollback()
        raise
    return alert.id


async def update_alert_status(
    session,
    alert_id: int,
    new_status: str,
    confirmed_at: Optional[datetime] = None,
) -> Optional[int]:
    """
    Update an alert's status (PENDING_REVIEW → CONFIRMED or DISMISSED).
    Returns the updated alert id, or None if alert not found.
    """
    from datetime import datetime as dt

    update_sql = text(
        """
        UPDATE alerts
        SET status = :status, confirmed_at = :confirmed_at
        WHERE id = :alert_id
        RETURNING id, status
        """
    )

    result = await session.execute(
        update_sql,
        {
            "alert_id": alert_id,
            "status": new_status,
            "confirmed_at": confirmed_at,
        },
    )
    row = result.fetchone()
    if row:
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        return row[0]
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return None
