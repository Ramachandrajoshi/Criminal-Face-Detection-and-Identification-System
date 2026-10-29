"""
Storage and ANN search for pose-tagged embeddings (``face_pose_embeddings``).

Each face profile can carry several embeddings taken at different head poses.
Search compares a query against all of them in one SQL statement and keeps
each person's closest pose, so a CCTV frame at a three-quarter angle matches
the person's three-quarter enrollment rather than their frontal mugshot.

All pose embeddings are InsightFace (embedding version 4).
"""

from typing import Optional

import numpy as np
from sqlalchemy import text

from app.core.config import INSIGHTFACE_EMBEDDING_VERSION
from app.core.crypto import encrypt_embedding_vector
from app.core.pose import HeadPose


def _vec_literal(vector: np.ndarray) -> str:
    return f"[{','.join(f'{v:.10f}' for v in vector.tolist())}]"


async def _insert_pose(session, face_id: int, embedding: np.ndarray, pose: HeadPose,
                       quality: float, tenant_id: int) -> int:
    vector = np.asarray(embedding, dtype=np.float32)
    result = await session.execute(
        text(
            """
            INSERT INTO face_pose_embeddings
                (face_id, tenant_id, embedding, embedding_enc, embedding_version,
                 pose_bucket, yaw, pitch, roll, quality)
            VALUES
                (:face_id, :tenant_id, CAST(:vec AS vector), :enc, :embedding_version,
                 :pose_bucket, :yaw, :pitch, :roll, :quality)
            RETURNING id
            """
        ),
        {
            "face_id": face_id,
            "tenant_id": tenant_id,
            "vec": _vec_literal(vector),
            "enc": encrypt_embedding_vector(vector),
            "embedding_version": INSIGHTFACE_EMBEDDING_VERSION,
            "pose_bucket": pose.bucket,
            "yaw": round(pose.yaw, 2),
            "pitch": round(pose.pitch, 2),
            "roll": round(pose.roll, 2),
            "quality": round(float(quality), 4),
        },
    )
    return int(result.scalar_one())


async def register_profile_with_poses(
    session,
    face_name: str,
    alias: Optional[str],
    demographics: Optional[dict],
    primary_embedding: np.ndarray,
    poses: list[tuple[np.ndarray, HeadPose, float]],
    tenant_id: int = 1,
) -> tuple[int, list[int]]:
    """
    Create a face profile and its pose embeddings in one transaction.

    ``primary_embedding`` (the average of the accepted photos) becomes the profile's own
    ``face_embedding`` as an InsightFace (version 4) vector, so the profile is
    also visible to the standard ``/search`` endpoint.

    Returns (profile_id, pose_ids).
    """
    vector = np.asarray(primary_embedding, dtype=np.float32)
    try:
        result = await session.execute(
            text(
                """
                INSERT INTO face_profiles
                    (face_name, alias, demographics, face_embedding, face_embedding_enc,
                     tenant_id, embedding_version)
                VALUES
                    (:face_name, :alias, :demographics, CAST(:vec AS vector), :enc,
                     :tenant_id, :embedding_version)
                RETURNING id
                """
            ),
            {
                "face_name": face_name,
                "alias": alias,
                "demographics": demographics,
                "vec": _vec_literal(vector),
                "enc": encrypt_embedding_vector(vector),
                "tenant_id": tenant_id,
                "embedding_version": INSIGHTFACE_EMBEDDING_VERSION,
            },
        )
        profile_id = int(result.scalar_one())
        pose_ids = [
            await _insert_pose(session, profile_id, emb, pose, quality, tenant_id)
            for emb, pose, quality in poses
        ]
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return profile_id, pose_ids


async def add_poses(
    session,
    face_id: int,
    poses: list[tuple[np.ndarray, HeadPose, float]],
    tenant_id: int = 1,
) -> list[int]:
    """Attach pose embeddings to an existing profile in one transaction."""
    try:
        pose_ids = [
            await _insert_pose(session, face_id, emb, pose, quality, tenant_id)
            for emb, pose, quality in poses
        ]
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return pose_ids


async def replace_poses(
    session,
    face_id: int,
    primary_embedding: np.ndarray,
    poses: list[tuple[np.ndarray, HeadPose, float]],
    tenant_id: int = 1,
) -> Optional[list[int]]:
    """
    Re-enrol an existing profile from a new set of poses in one transaction.

    The profile's old pose embeddings are deleted and ``poses`` stored in
    their place. ``primary_embedding`` replaces the profile's own
    ``face_embedding`` as a version 4 vector, as for a new profile in
    ``register_profile_with_poses``. Name, alias and demographics are kept.

    Returns the new pose ids, or None when the profile does not exist (nothing
    is changed).
    """
    vector = np.asarray(primary_embedding, dtype=np.float32)
    try:
        # Updating the profile first locks its row, so two re-enrolments of the
        # same face run one after the other instead of mixing their poses.
        result = await session.execute(
            text(
                """
                UPDATE face_profiles
                SET face_embedding     = CAST(:vec AS vector),
                    face_embedding_enc = :enc,
                    embedding_version  = :embedding_version
                WHERE id = :face_id AND tenant_id = :tenant_id
                RETURNING id
                """
            ),
            {
                "face_id": face_id,
                "tenant_id": tenant_id,
                "vec": _vec_literal(vector),
                "enc": encrypt_embedding_vector(vector),
                "embedding_version": INSIGHTFACE_EMBEDDING_VERSION,
            },
        )
        if result.fetchone() is None:
            await session.rollback()
            return None
        await session.execute(
            text(
                """
                DELETE FROM face_pose_embeddings
                WHERE face_id = :face_id AND tenant_id = :tenant_id
                """
            ),
            {"face_id": face_id, "tenant_id": tenant_id},
        )
        pose_ids = [
            await _insert_pose(session, face_id, emb, pose, quality, tenant_id)
            for emb, pose, quality in poses
        ]
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return pose_ids


async def count_poses(session, face_id: int, tenant_id: int = 1) -> int:
    result = await session.execute(
        text(
            """
            SELECT COUNT(*) FROM face_pose_embeddings
            WHERE face_id = :face_id AND tenant_id = :tenant_id
            """
        ),
        {"face_id": face_id, "tenant_id": tenant_id},
    )
    return int(result.scalar_one())


async def list_poses(session, face_id: int, tenant_id: int = 1) -> list[dict]:
    """Pose metadata for a profile. Embeddings are never returned."""
    result = await session.execute(
        text(
            """
            SELECT id, pose_bucket, yaw, pitch, roll, quality, created_at
            FROM face_pose_embeddings
            WHERE face_id = :face_id AND tenant_id = :tenant_id
            ORDER BY id
            """
        ),
        {"face_id": face_id, "tenant_id": tenant_id},
    )
    def _r(value, digits):
        # REAL columns come back as float32 noise (1.4600000381…).
        return None if value is None else round(float(value), digits)

    return [
        {
            "id": row[0],
            "pose_bucket": row[1],
            "yaw": _r(row[2], 2),
            "pitch": _r(row[3], 2),
            "roll": _r(row[4], 2),
            "quality": _r(row[5], 4),
            "created_at": row[6].isoformat() if row[6] else "",
        }
        for row in result
    ]


async def delete_pose(session, face_id: int, pose_id: int, tenant_id: int = 1) -> bool:
    """Delete one pose embedding. Returns False when it does not exist."""
    result = await session.execute(
        text(
            """
            DELETE FROM face_pose_embeddings
            WHERE id = :pose_id AND face_id = :face_id AND tenant_id = :tenant_id
            RETURNING id
            """
        ),
        {"pose_id": pose_id, "face_id": face_id, "tenant_id": tenant_id},
    )
    row = result.fetchone()
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return row is not None


async def pose_ann_query(
    session,
    query_vector: np.ndarray,
    threshold: float,
    limit: int = 10,
    tenant_id: int = 1,
) -> list[dict]:
    """
    Closest pose per person, ranked by cosine distance — one SQL statement.

    ``DISTINCT ON (face_id)`` keeps each person's best-matching pose, so a
    person enrolled at five angles appears once, scored by the angle nearest
    the query.
    """
    result = await session.execute(
        text(
            """
            SELECT face_id, face_name, alias, distance, pose_bucket, yaw, pitch
            FROM (
                SELECT DISTINCT ON (p.face_id)
                    p.face_id,
                    f.face_name,
                    f.alias,
                    (p.embedding <=> CAST(:vec AS vector)) AS distance,
                    p.pose_bucket,
                    p.yaw,
                    p.pitch
                FROM face_pose_embeddings p
                JOIN face_profiles f ON f.id = p.face_id AND f.tenant_id = p.tenant_id
                WHERE p.tenant_id = :tenant_id
                  AND p.embedding_version = :embedding_version
                  AND (p.embedding <=> CAST(:vec AS vector)) <= :threshold
                ORDER BY p.face_id, distance ASC
            ) best_per_face
            ORDER BY distance ASC
            LIMIT :limit
            """
        ),
        {
            "vec": _vec_literal(np.asarray(query_vector, dtype=np.float32)),
            "threshold": threshold,
            "limit": limit,
            "tenant_id": tenant_id,
            "embedding_version": INSIGHTFACE_EMBEDDING_VERSION,
        },
    )
    return [
        {
            "id": row[0],
            "face_name": row[1],
            "alias": row[2],
            "distance": round(float(row[3]), 6),
            "matched_pose": row[4],
            "matched_yaw": None if row[5] is None else round(float(row[5]), 2),
            "matched_pitch": None if row[6] is None else round(float(row[6]), 2),
            "tenant_id": tenant_id,
        }
        for row in result
    ]
