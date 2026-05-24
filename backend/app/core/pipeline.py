"""
Core pipeline — five-stage face processing pipeline.

Stage 1 — DETECT   : OpenCV frame grab → RetinaFace bounding box
Stage 2 — ALIGN    : Eye-landmark affine transform
Stage 3 — NORMALIZE: Resize & pixel normalisation
Stage 4 — REPRESENT: deepface.represent() → 512-d ArcFace embedding
Stage 5 — VERIFY   : pgvector cosine ANN query

Anti-spoofing: Liveness check runs after detection, before embedding extraction.
If liveness fails, pipeline returns SPOOF_BLOCKED and logs to audit.
"""

import io
import logging
from typing import Optional

import numpy as np
from fastapi import UploadFile

from app.core.config import settings
from app.db.vector_ops import compute_query_hash

logger = logging.getLogger(__name__)


# ---------- Stage 1: DETECT ----------
def detect_face(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Detect and extract the largest face from image bytes.
    Returns aligned numpy array (RGB) or None if no face found.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("Failed to decode image bytes")
        return None

    # Detector priority: retinaface → mtcnn → opencv
    detector_backend = settings.deepface_detector
    from deepface import DeepFace

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )
    except Exception as exc:
        logger.warning("Detector %s failed: %s — trying mtcnn fallback", detector_backend, exc)
        try:
            from deepface import DeepFace
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="mtcnn",
                enforce_detection=False,
                align=True,
            )
        except Exception as exc2:
            logger.error("MTCNN fallback also failed: %s", exc2)
            return None

    if not faces:
        logger.warning("No face detected in frame")
        return None

    # Pick the face with highest confidence (first result)
    face = faces[0]
    aligned = face["face"]
    return aligned


# ---------- Stage 2-3: ALIGN + NORMALIZE ----------
# These are handled internally by DeepFace.extract_faces(align=True).
# No additional code needed.


# ---------- Stage 4: REPRESENT ----------
def extract_embedding(aligned_face: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a 512-d ArcFace embedding from an aligned face image.
    Returns numpy array or None on failure.
    """
    from deepface import DeepFace

    try:
        result = DeepFace.represent(
            img_path=aligned_face,
            model_name=settings.deepface_model,
            detector_backend=settings.deepface_detector,
            enforce_detection=False,
        )
    except Exception as exc:
        logger.error("Embedding extraction failed: %s", exc)
        return None

    if not result:
        logger.error("Empty representation returned")
        return None

    embedding = np.array(result[0]["embedding"], dtype=np.float32)

    # L2 normalise
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ---------- Stage 5: VERIFY (ANN query) ----------
async def verify_match(
    embedding: np.ndarray,
    session,
    threshold: Optional[float] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Execute pgvector ANN query with the given embedding.
    Returns list of match dicts (sorted by distance).
    """
    from app.db.vector_ops import cosine_ann_query, compute_query_hash

    query_hash = compute_query_hash(embedding.tobytes())
    query_list = embedding.tolist()

    matches = await cosine_ann_query(session, query_list, threshold=threshold, limit=limit)

    # Attach query_hash to each match for audit trail
    for m in matches:
        m["query_hash"] = query_hash

    return matches


# ---------- Liveness Check ----------
def check_liveness_on_bytes(image_bytes: bytes) -> bool:
    """
    Run anti-spoofing liveness check on raw image bytes.
    Returns True if the image passes liveness (real person),
    False if a spoof is detected.

    Integrated into the pipeline to comply with AGENTS.md §7.
    """
    from app.core.liveness import check_liveness_with_deepface
    result = check_liveness_with_deepface(image_bytes)
    return result.get("is_live", False)


# ---------- Full Pipeline ----------
async def run_pipeline(
    file: UploadFile,
    session,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    suspect_name: Optional[str] = None,
) -> dict:
    """
    Execute the full five-stage pipeline on an uploaded image.

    Returns a dict with:
      - status: "MATCH" | "NO_MATCH" | "SPOOF_BLOCKED" | "ERROR"
      - matches: list of match dicts (empty for NO_MATCH)
      - query_hash: SHA-256 of the embedding
    """
    import hashlib

    content = await file.read()

    # Anti-spoofing: check liveness before proceeding
    if check_liveness_on_bytes(content):
        aligned = detect_face(content)
    else:
        # Liveness check failed — block and log
        query_hash = compute_query_hash(b"")
        await _add_spoof_audit(session, query_hash, content)
        return {
            "status": "SPOOF_BLOCKED",
            "matches": [],
            "query_hash": query_hash,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
        }

    if aligned is None:
        return {"status": "ERROR", "error": "No face detected", "matches": []}

    # Stage 4: Represent
    embedding = extract_embedding(aligned)
    if embedding is None:
        return {"status": "ERROR", "error": "Embedding extraction failed", "matches": []}

    query_hash = compute_query_hash(embedding.tobytes())

    # Stage 5: Verify — ANN search
    matches = await verify_match(embedding, session)

    if matches:
        best = matches[0]
        return {
            "status": "MATCH",
            "matches": matches,
            "query_hash": query_hash,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
        }
    else:
        return {
            "status": "NO_MATCH",
            "matches": [],
            "query_hash": query_hash,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
        }


async def _add_spoof_audit(session, query_hash: str, image_bytes: bytes):
    """Helper: log a SPOOF_BLOCKED audit entry."""
    from app.db.vector_ops import add_audit_entry
    await add_audit_entry(
        session,
        event_type="SPOOF_BLOCKED",
        query_hash=query_hash,
    )


async def register_pipeline(
    file: UploadFile,
    session,
    suspect_name: str,
    alias: Optional[str] = None,
    demographics: Optional[dict] = None,
) -> dict:
    """
    Register a new suspect: stages 1-4 + DB insert.
    Returns profile id and embedding.
    """
    content = await file.read()

    # Anti-spoofing: check liveness before registering
    if check_liveness_on_bytes(content):
        aligned = detect_face(content)
    else:
        query_hash = compute_query_hash(b"")
        await _add_spoof_audit(session, query_hash, content)
        return {
            "status": "SPOOF_BLOCKED",
            "error": "Liveness check failed — possible spoof detected",
        }

    if aligned is None:
        return {"status": "ERROR", "error": "No face detected"}

    embedding = extract_embedding(aligned)
    if embedding is None:
        return {"status": "ERROR", "error": "Embedding extraction failed"}

    from app.db.vector_ops import register_profile, add_audit_entry

    profile_id = await register_profile(
        session, suspect_name, alias, demographics, embedding.tobytes()
    )

    query_hash = compute_query_hash(embedding.tobytes())
    await add_audit_entry(
        session,
        event_type="REGISTER",
        query_hash=query_hash,
        result_name=suspect_name,
    )

    return {
        "status": "REGISTERED",
        "profile_id": profile_id,
        "query_hash": query_hash,
        "embedding_dim": int(embedding.shape[0]),
    }
