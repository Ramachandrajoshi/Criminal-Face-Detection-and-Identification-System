"""
pgvector ANN query helpers.
Handles AES-256 decryption of stored embeddings before cosine similarity search.
"""

import hashlib
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
    tenant_id: int = 1,
) -> list[dict]:
    """
    Perform approximate nearest-neighbour cosine similarity search
    using the pgvector HNSW index.

    Uses the pgvector `<=>` operator which returns COSINE DISTANCE
    (0 = identical, 2 = opposite). Lower distance = better match.

    Returns a list of dicts: {id, person_name, alias, distance, ...}
    sorted by ascending distance (closest match first).

    threshold defaults to MATCH_THRESHOLD from config.
    tenant_id filters results to the requesting tenant (defense in depth
    alongside PostgreSQL RLS).
    """
    threshold = threshold if threshold is not None else settings.match_threshold

    if isinstance(query_vector, (bytes, bytearray)):
        query_vector = decrypt_embedding_bytes(bytes(query_vector)).tolist()

    # Build vector literal from list for pgvector
    vec_str = f"[{','.join(f'{v:.10f}' for v in query_vector)}]"

    # Cosine distance via <=> operator (lower is better match)
    # Filter: distance <= threshold (e.g., 0.58) AND tenant_id match
    sql = text(
        """
        SELECT
            id,
            face_name,
            alias,
            (face_embedding <=> :vec) AS distance,
            embedding_version
        FROM face_profiles
        WHERE (face_embedding <=> :vec) <= :threshold
          AND tenant_id = :tenant_id
        ORDER BY distance ASC
        LIMIT :limit
        """
    )

    result = await session.execute(
        sql,
        {"vec": vec_str, "threshold": threshold, "limit": limit, "tenant_id": tenant_id},
    )

    rows = []
    for row in result:
        rows.append({
            "id": row[0],
            "face_name": row[1],
            "alias": row[2],
            "distance": round(float(row[3]), 6),
            "embedding_version": row[4],
            "tenant_id": tenant_id,
        })

    return rows


async def register_profile(
    session,
    person_name: str,
    alias: Optional[str],
    demographics: Optional[dict],
    embedding: np.ndarray | list[float],
    tenant_id: int = 1,
) -> int:
    """
    Insert a new face profile.
    Embeddings are stored as pgvector plus AES-256 encrypted payload at rest.
    """
    vector = np.asarray(embedding, dtype=np.float32)
    vec_str = f"[{','.join(f'{v:.10f}' for v in vector.tolist())}]"
    encrypted = encrypt_embedding_vector(vector)

    sql = text(
        """
        INSERT INTO face_profiles (face_name, alias, demographics, face_embedding, face_embedding_enc, tenant_id, embedding_version)
        VALUES (:face_name, :alias, :demographics, CAST(:vec AS vector), :enc, :tenant_id, 2)
        RETURNING id
        """
    )

    result = await session.execute(
        sql,
        {
            "face_name": person_name,
            "alias": alias,
            "demographics": demographics,
            "vec": vec_str,
            "enc": encrypted,
            "tenant_id": tenant_id,
        },
    )

    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return int(result.scalar_one())


async def update_face_embedding(
    session,
    face_id: int,
    embedding: "np.ndarray | list[float]",
    tenant_id: int = 1,
) -> bool:
    """
    Replace the face_embedding (pgvector column) and face_embedding_enc
    (AES-256 encrypted payload) for an existing face profile.

    Returns True when a row was updated, False when face_id was not found.
    Raises on database errors (caller is responsible for rollback).
    """
    vector = np.asarray(embedding, dtype=np.float32)
    vec_str = f"[{','.join(f'{v:.10f}' for v in vector.tolist())}]"
    encrypted = encrypt_embedding_vector(vector)

    sql = text(
        """
        UPDATE face_profiles
        SET face_embedding     = CAST(:vec AS vector),
            face_embedding_enc = :enc,
            embedding_version  = 2
        WHERE id = :face_id AND tenant_id = :tenant_id
        RETURNING id
        """
    )

    result = await session.execute(
        sql,
        {
            "face_id": face_id,
            "vec": vec_str,
            "enc": encrypted,
            "tenant_id": tenant_id,
        },
    )
    row = result.fetchone()

    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return row is not None


async def add_audit_entry(
    session,
    event_type: str,
    query_hash: str,
    result_name: Optional[str] = None,
    distance: Optional[float] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    tenant_id: int = 1,
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
        tenant_id=tenant_id,
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
    face_id: Optional[int],
    event_type: str,
    distance: Optional[float],
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    tenant_id: int = 1,
) -> int:
    """Create an alert for human-in-the-loop review."""
    from app.db.models import Alert

    alert = Alert(
        audit_log_id=audit_log_id,
        face_id=face_id,
        event_type=event_type,
        distance=distance,
        status="PENDING_REVIEW",
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        tenant_id=tenant_id,
    )
    session.add(alert)
    try:
        await session.commit()
        await session.refresh(alert)
    except Exception:
        await session.rollback()
        raise
    return alert.id



