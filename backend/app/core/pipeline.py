"""
Core pipeline — five-stage face processing pipeline.

Stage 1 — DETECT   : OpenCV frame grab → RetinaFace bounding box
Stage 2 — ALIGN    : Eye-landmark affine transform
Stage 3 — NORMALIZE: Resize & pixel normalisation
Stage 4 — REPRESENT: deepface.represent() → 512-d ArcFace embedding
Stage 5 — VERIFY   : pgvector cosine ANN query

Liveness / Anti-spoofing policy
---------------------------------
Liveness detection is only meaningful for **live camera captures**.  It works
by detecting physiological cues (blinks, texture gradients) that a printed
photograph cannot exhibit.  Applying it to uploaded image files always fails
because a static JPEG has no such cues — every uploaded image looks like a
"spoof" to the classifier.

Therefore:
  * ``register_pipeline`` — no liveness check.  Registration always uses
    uploaded photos, not live captures.
  * ``run_pipeline`` (search) — liveness is checked **only** when
    ``enforce_liveness=True`` is passed, which the search route sets only for
    live camera input.  Photo uploads pass ``enforce_liveness=False``.
"""

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

    Detector priority (AGENTS.md §5): retinaface → mtcnn → opencv.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("Failed to decode image bytes")
        return None

    from deepface import DeepFace

    for backend in [settings.deepface_detector, "mtcnn", "opencv"]:
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=backend,
                enforce_detection=False,
                align=True,
            )
            if faces:
                return faces[0]["face"]
        except Exception as exc:
            logger.warning("Detector %s failed: %s — trying next backend", backend, exc)

    logger.warning("No face detected in frame after all fallbacks")
    return None


# ---------- Stage 2-3: ALIGN + NORMALIZE ----------
# Handled internally by DeepFace.extract_faces(align=True).


# ---------- Stage 4: REPRESENT ----------
def extract_embedding(aligned_face: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a 512-d ArcFace embedding from an aligned face image.
    Returns L2-normalised numpy array or None on failure.
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

    # L2 normalise (Stage 3 normalisation — AGENTS.md §3)
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
    Execute pgvector cosine ANN query with the given embedding.
    Returns list of match dicts sorted by ascending distance.
    """
    from app.db.vector_ops import cosine_ann_query

    query_hash = compute_query_hash(embedding.tobytes())
    matches = await cosine_ann_query(session, embedding.tolist(), threshold=threshold, limit=limit)

    for m in matches:
        m["query_hash"] = query_hash

    return matches


# ---------- Liveness Check ----------
def check_liveness_on_bytes(image_bytes: bytes) -> bool:
    """
    Run DeepFace anti-spoofing on raw image bytes.

    Only call this for **live camera captures**.  Do NOT call it for uploaded
    photo files — static images always fail liveness and would produce false
    SPOOF_BLOCKED results.
    """
    from app.core.liveness import check_liveness_with_deepface
    result = check_liveness_with_deepface(image_bytes)
    return result.get("is_live", False)


# ---------- Audit helpers ----------
async def _add_spoof_audit(
    session,
    query_hash: str,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
) -> None:
    """Append a SPOOF_BLOCKED entry to the immutable audit log."""
    from app.db.vector_ops import add_audit_entry
    await add_audit_entry(
        session,
        event_type="SPOOF_BLOCKED",
        query_hash=query_hash,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )


# ---------- Search Pipeline ----------
async def run_pipeline(
    file: UploadFile,
    session,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    limit: int = 10,
    enforce_liveness: bool = False,
) -> dict:
    """
    Execute the full five-stage detection pipeline on an uploaded image.

    Parameters
    ----------
    enforce_liveness:
        Set ``True`` **only** for live camera captures.  When ``False``
        (default) the liveness check is skipped entirely so that uploaded
        photos are processed without triggering false SPOOF_BLOCKED results.

    Returns a dict with keys:
      status       — "MATCH" | "NO_MATCH" | "SPOOF_BLOCKED" | "ERROR"
      matches      — list of match dicts (empty for NO_MATCH / SPOOF_BLOCKED)
      query_hash   — SHA-256 of the embedding (or empty-bytes hash on block)
    """
    content = await file.read()

    # ── Liveness gate (camera only) ───────────────────────────────
    if enforce_liveness:
        if not check_liveness_on_bytes(content):
            query_hash = compute_query_hash(b"")
            await _add_spoof_audit(session, query_hash, gps_lat=gps_lat, gps_lon=gps_lon)
            return {
                "status": "SPOOF_BLOCKED",
                "matches": [],
                "query_hash": query_hash,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
            }

    # ── Stage 1: Detect ───────────────────────────────────────────
    aligned = detect_face(content)
    if aligned is None:
        return {"status": "ERROR", "error": "No face detected", "matches": []}

    # ── Stage 4: Represent ────────────────────────────────────────
    embedding = extract_embedding(aligned)
    if embedding is None:
        return {"status": "ERROR", "error": "Embedding extraction failed", "matches": []}

    query_hash = compute_query_hash(embedding.tobytes())

    # ── Stage 5: Verify ───────────────────────────────────────────
    matches = await verify_match(embedding, session, limit=limit)

    if matches:
        return {
            "status": "MATCH",
            "matches": matches,
            "query_hash": query_hash,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
        }
    return {
        "status": "NO_MATCH",
        "matches": [],
        "query_hash": query_hash,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
    }


# ---------- Registration Pipeline ----------
async def register_pipeline(
    file: UploadFile,
    session,
    suspect_name: str,
    alias: Optional[str] = None,
    demographics: Optional[dict] = None,
) -> dict:
    """
    Register a new suspect: stages 1–4 + database insert.

    **No liveness check is performed.**  Registration always uses uploaded
    photo files (mugshots, ID photos, CCTV stills) which are static images.
    Liveness detection is meaningless for static images — see module docstring.

    Returns a dict with keys: status, profile_id, query_hash, embedding_dim.
    """
    content = await file.read()

    # Stage 1: Detect
    aligned = detect_face(content)
    if aligned is None:
        return {"status": "ERROR", "error": "No face detected in the uploaded image"}

    # Stage 4: Represent
    embedding = extract_embedding(aligned)
    if embedding is None:
        return {"status": "ERROR", "error": "Embedding extraction failed"}

    from app.db.vector_ops import add_audit_entry, register_profile

    profile_id = await register_profile(session, suspect_name, alias, demographics, embedding)

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
