"""
pgvector ANN query helpers.
Handles AES-256 decryption of stored embeddings before cosine similarity search.
"""

import hashlib
from typing import Optional

import numpy as np
from sqlalchemy import text

from app.core.config import CURRENT_EMBEDDING_VERSION, match_threshold_for_version
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
    target_version: Optional[int] = None,
) -> list[dict]:
    """
    Perform approximate nearest-neighbour cosine similarity search
    using the pgvector HNSW index.

    Uses the pgvector `<=>` operator which returns COSINE DISTANCE
    (0 = identical, 2 = opposite). Lower distance = better match.

    Returns a list of dicts: {id, person_name, alias, distance, ...}
    sorted by ascending distance (closest match first).

    threshold defaults to the configured threshold for ``target_version``
    (MATCH_THRESHOLD for versions 1–3, MATCH_THRESHOLD_INSIGHTFACE for 4).
    tenant_id filters results to the requesting tenant (defense in depth
    alongside PostgreSQL RLS).
    """
    if threshold is None:
        threshold = match_threshold_for_version(
            target_version if target_version is not None else CURRENT_EMBEDDING_VERSION
        )

    if isinstance(query_vector, (bytes, bytearray)):
        query_vector = decrypt_embedding_bytes(bytes(query_vector)).tolist()

    # Build vector literal from list for pgvector
    vec_str = f"[{','.join(f'{v:.10f}' for v in query_vector)}]"

    # Cosine distance via <=> operator (lower is better match)
    # Filter: distance <= threshold AND tenant_id match AND (optional) version match
    version_filter = "AND embedding_version = :target_version" if target_version is not None else ""
    sql = text(
        f"""
        SELECT
            id,
            face_name,
            alias,
            (face_embedding <=> :vec) AS distance,
            embedding_version
        FROM face_profiles
        WHERE (face_embedding <=> :vec) <= :threshold
          AND tenant_id = :tenant_id
          {version_filter}
        ORDER BY distance ASC
        LIMIT :limit
        """
    )

    result = await session.execute(
        sql,
        {
            "vec": vec_str, 
            "threshold": threshold, 
            "limit": limit, 
            "tenant_id": tenant_id,
            "target_version": target_version
        },
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


async def get_present_embedding_versions(session, tenant_id: int = 1) -> set[int]:
    """
    Return the set of ``embedding_version`` values that actually have rows for
    this tenant.

    Search embeds the query once per recipe, which is the expensive part of a
    lookup. Recipes with no enrolled rows can never produce a match, so this
    lets the pipeline skip embedding them entirely — the common case after a
    version bump, when every profile still carries the previous recipe.

    Backed by the ``face_profiles_tenant_version_idx`` index, so this is an
    index-only scan rather than a table scan.
    """
    sql = text(
        """
        SELECT DISTINCT embedding_version
        FROM face_profiles
        WHERE tenant_id = :tenant_id
        """
    )
    result = await session.execute(sql, {"tenant_id": tenant_id})
    return {int(row[0]) for row in result if row[0] is not None}


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
        VALUES (:face_name, :alias, :demographics, CAST(:vec AS vector), :enc, :tenant_id, :embedding_version)
        RETURNING id
        """
    )

    result = await session.execute(
        sql,
        {
            "face_name": person_name,
            "alias": alias,
            "demographics": demographics,
            "embedding_version": CURRENT_EMBEDDING_VERSION,
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
            embedding_version  = :embedding_version
        WHERE id = :face_id AND tenant_id = :tenant_id
        RETURNING id
        """
    )

    result = await session.execute(
        sql,
        {
            "face_id": face_id,
            "embedding_version": CURRENT_EMBEDDING_VERSION,
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



