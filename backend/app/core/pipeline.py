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


# Minimum detector confidence to accept a bounding box as a real face.
# Detections below this threshold are treated as "no face found".
# Tune this in config if the primary detector is too aggressive.
_MIN_FACE_CONFIDENCE = 0.70


# ---------- Stage 1: DETECT ----------
def detect_face(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Detect and extract the largest face from image bytes.
    Returns aligned numpy array (RGB) or None if no face found.

    Detector priority (AGENTS.md §5): retinaface → mtcnn → opencv.

    enforce_detection=True
    ----------------------
    DeepFace raises ``ValueError`` when no face bounding box meets the
    detector's internal threshold.  We catch that specifically so we can
    silently fall through to the next backend.  Only non-ValueError
    exceptions (driver failures, OOM, etc.) are logged as warnings.

    Confidence filter
    -----------------
    Even with enforce_detection=True some detectors return very low-
    confidence boxes on flag icons, logos, or blurry thumbnails.  We
    discard any detection whose confidence is below ``_MIN_FACE_CONFIDENCE``
    (0.70) as an extra guard, then try the next backend.
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
                enforce_detection=True,   # raises ValueError if no face found
                align=True,
            )

            # Filter out low-confidence bounding boxes (e.g. flag icons, logos).
            confident_faces = [
                f for f in faces
                if f.get("confidence", 1.0) >= _MIN_FACE_CONFIDENCE
            ]

            if confident_faces:
                # Return the largest face by bounding-box area.
                best = max(
                    confident_faces,
                    key=lambda f: (
                        f.get("facial_area", {}).get("w", 0)
                        * f.get("facial_area", {}).get("h", 0)
                    ),
                )
                logger.debug(
                    "Face detected by %s — confidence=%.2f, area=%dx%d",
                    backend,
                    best.get("confidence", 0),
                    best.get("facial_area", {}).get("w", 0),
                    best.get("facial_area", {}).get("h", 0),
                )
                return best["face"]

            logger.debug(
                "Backend %s found %d face(s) but all below confidence threshold %.2f — trying next",
                backend, len(faces), _MIN_FACE_CONFIDENCE,
            )

        except ValueError:
            # DeepFace raises ValueError("Face could not be detected ...") when
            # no face bounding box is found with enforce_detection=True.
            # This is the expected path for images without faces — not an error.
            logger.debug("No face bounding box found by detector '%s' — trying next", backend)
        except Exception as exc:
            # Real driver / model failures (e.g. OOM, missing weights).
            logger.warning("Detector '%s' errored: %s — trying next backend", backend, exc)

    logger.warning(
        "No face detected in image after trying all detector backends %s. "
        "Image will be marked as FAILED.",
        [settings.deepface_detector, "mtcnn", "opencv"],
    )
    return None


# ---------- Stage 2-3: ALIGN + NORMALIZE ----------

def _preprocess_for_arcface(face: np.ndarray) -> np.ndarray:
    """
    Prepare an aligned face array for ArcFace embedding extraction.

    Input contract
    --------------
    ``face`` is the array returned by ``DeepFace.extract_faces(align=True)``.
    DeepFace stores it as **float32 in [0, 1]** (it divided by 255 internally).
    ArcFace's training normalization is:

        pixel_norm = (pixel_uint8 − 127.5) / 128

    which expects **uint8 [0, 255]** values.  Without this conversion the
    model receives near-zero inputs (÷255 twice → [0, 0.004]) that are far
    outside the training distribution — the main cause of inflated
    same-person cosine distances.

    Steps
    -----
    All enhancement steps operate on the *large* crop (e.g. 300×400 px from
    the detector) BEFORE the final 112×112 resize, so the quality gain carries
    through to the embedding.

    0. uint8 conversion     — restore [0, 255] if input is float32 from DeepFace.
    1. Upscale tiny crops   — Lanczos super-sample if shortest side < threshold.
                              CCTV sub-crops are often < 80 px; upsampling before
                              any enhancement gives CLAHE and the sharpener more
                              pixels to work with.
    2. Denoising            — fastNlMeansDenoisingColored (h=5, hColor=5).
                              Applied BEFORE CLAHE so the equaliser amplifies
                              signal rather than sensor noise.
    3. Auto-gamma           — gamma = log(0.5) / log(mean_luminance / 255).
                              Lifts dark / under-exposed faces toward mid-tone.
                              For already-bright images gamma ≈ 1.0 (no-op).
    4. CLAHE                — Contrast Limited Adaptive Histogram Equalization
                              on the LAB L* channel.  Normalises local illumination
                              without altering hue or saturation, tightening
                              same-person embedding variance across lighting.
    5. Unsharp mask         — Recovers edge detail blurred by the denoiser.
                              output = strength × sharp − (strength−1) × blurred
    6. Pad → 112×112 Lanczos resize — ArcFace canonical size.
    """
    import cv2
    import math

    # ── Step 0: uint8 conversion ─────────────────────────────────────────────
    if face.dtype != np.uint8:
        face = (face * 255.0).clip(0, 255).astype(np.uint8)

    # At this point face is RGB uint8.
    # Convert to BGR once for OpenCV processing; we'll convert back at the end.
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    # ── Step 1: Upscale tiny crops ───────────────────────────────────────────
    # CCTV face crops can be as small as 50×50 px. Upscaling to at least
    # `small_crop_min_side` before enhancement gives the subsequent filters
    # significantly more pixels to work with and reduces blocking artefacts.
    if settings.enable_upscale_small_crops:
        h, w = face_bgr.shape[:2]
        min_side = min(h, w)
        if min_side < settings.small_crop_min_side:
            scale = settings.small_crop_min_side / min_side
            new_w = max(int(w * scale), settings.small_crop_min_side)
            new_h = max(int(h * scale), settings.small_crop_min_side)
            face_bgr = cv2.resize(
                face_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )
            logger.debug(
                "Upscaled small crop from %dx%d → %dx%d (scale=%.2f)",
                w, h, new_w, new_h, scale,
            )

    # ── Step 2: Denoising ────────────────────────────────────────────────────
    # Non-local means denoising removes CCTV sensor/compression noise before
    # the contrast equaliser runs. h=5 is conservative — strong enough to
    # smooth noise without blurring eye or lip edges.
    if settings.enable_denoising:
        face_bgr = cv2.fastNlMeansDenoisingColored(
            face_bgr,
            None,
            h=5,
            hColor=5,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # ── Step 3: Auto-gamma correction ────────────────────────────────────────
    # Computes a per-image gamma that maps the observed mean luminance toward
    # the perceptual mid-tone (0.5 in [0,1]) using the relation:
    #
    #   γ = log(0.5) / log(μ / 255)
    #
    # The LUT applies x^γ (not x^(1/γ)):
    #   • Dark image  (μ ≈ 40):  γ ≈ 0.32  → x^0.32  lifts shadows  ✓
    #   • Mid image   (μ ≈ 128): γ ≈ 1.0   → x^1.0   identity       ✓
    #   • Bright image(μ ≈ 220): γ ≈ 4.5   → x^4.5   compresses highlights ✓
    if settings.enable_gamma_correction:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        if mean_lum > 1.0:  # guard against pure-black frames
            gamma = math.log(0.5) / math.log(mean_lum / 255.0)
            gamma = float(np.clip(gamma, 0.1, 5.0))  # prevent extreme correction
            # Apply x^gamma: for dark images (gamma<1) this lifts shadows.
            lut = np.array(
                [((i / 255.0) ** gamma) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            face_bgr = cv2.LUT(face_bgr, lut)
            logger.debug(
                "Auto-gamma: mean_lum=%.1f → gamma=%.3f", mean_lum, gamma
            )


    # ── Step 4: CLAHE (LAB L* channel) ───────────────────────────────────────
    # Contrast-limited equalisation on the luminance channel only.
    # Operates AFTER gamma so the histogram being equalised is already
    # reasonably centred, letting CLAHE handle local contrast rather than
    # global exposure.
    if settings.enable_clahe:
        face_lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip_limit,
            tileGridSize=(8, 8),
        )
        face_lab[:, :, 0] = clahe.apply(face_lab[:, :, 0])
        face_bgr = cv2.cvtColor(face_lab, cv2.COLOR_LAB2BGR)

    # ── Step 5: Unsharp mask ─────────────────────────────────────────────────
    # Recovers the edge detail that fastNlMeansDenoisingColored softens.
    # Formula: output = strength × sharp − (strength − 1) × blurred
    # With strength=1.5: output = 1.5 × face − 0.5 × blurred  (+50% edges)
    # Uses cv2.addWeighted which clamps to uint8 automatically.
    if settings.enable_unsharp_mask:
        blurred = cv2.GaussianBlur(face_bgr, (0, 0), sigmaX=2.0)
        s = float(settings.unsharp_strength)
        face_bgr = cv2.addWeighted(face_bgr, s, blurred, -(s - 1.0), 0)

    # ── Step 6: Pad → 112×112 Lanczos resize ─────────────────────────────────
    # Convert back to RGB for ArcFace.  Pad to square first to avoid
    # stretching the face geometry (a direct 112×112 resize on a non-square
    # crop collapses inter-class distance — see original docstring).
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    if face.shape[:2] != (112, 112):
        h, w = face.shape[:2]
        side = max(h, w)
        top, bottom = divmod(side - h, 2)
        left, right = divmod(side - w, 2)
        face = cv2.copyMakeBorder(
            face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LANCZOS4)

    return face


# ---------- Stage 4: REPRESENT ----------
def extract_embedding(aligned_face: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a 512-d ArcFace embedding from an aligned face image.
    Returns L2-normalised numpy array or None on failure.

    Normalization note
    ------------------
    DeepFace.represent's ``normalization`` parameter must be set to
    ``"ArcFace"`` (i.e. ``(pixel − 127.5) / 128``) to match what the
    InsightFace/ArcFace backbone was trained with.  The DeepFace default
    (``"base"`` = pixel / 255) is correct only for VGGFace, NOT for ArcFace.
    Passing the wrong normalization produces embeddings that are statistically
    valid but not aligned with the model weights, increasing intra-class
    variance (= higher same-person distance).
    """
    from deepface import DeepFace

    # Run Stage 2-3 preprocessing before handing off to the ArcFace model.
    face = _preprocess_for_arcface(aligned_face)

    try:
        result = DeepFace.represent(
            img_path=face,
            model_name=settings.deepface_model,
            detector_backend="skip",       # already aligned + preprocessed
            enforce_detection=False,
            normalization=settings.arcface_normalization,   # "ArcFace" = (x-127.5)/128
        )
    except Exception as exc:
        logger.error("Embedding extraction failed: %s", exc)
        return None

    if not result:
        logger.error("Empty representation returned")
        return None

    embedding = np.array(result[0]["embedding"], dtype=np.float32)

    # L2 normalise for cosine distance (AGENTS.md §3)
    # DeepFace may already L2-normalise depending on the model; re-normalising
    # is idempotent and guarantees unit-norm vectors for pgvector cosine ops.
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
    tenant_id: Optional[int] = None,
) -> list[dict]:
    """
    Execute pgvector cosine ANN query with the given embedding.
    Returns list of match dicts sorted by ascending distance.
    """
    from app.db.vector_ops import cosine_ann_query

    query_hash = compute_query_hash(embedding.tobytes())
    matches = await cosine_ann_query(
        session,
        embedding.tolist(),
        threshold=threshold,
        limit=limit,
        tenant_id=tenant_id if tenant_id is not None else 1,
    )

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
    tenant_id: int = 1,
) -> None:
    """Append a SPOOF_BLOCKED entry to the immutable audit log."""
    from app.db.vector_ops import add_audit_entry
    await add_audit_entry(
        session,
        event_type="SPOOF_BLOCKED",
        query_hash=query_hash,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        tenant_id=tenant_id,
    )


# ---------- Search Pipeline ----------
async def run_pipeline(
    file: UploadFile,
    session,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    limit: int = 10,
    enforce_liveness: bool = False,
    tenant_id: int = 1,
) -> dict:
    """
    Execute the full five-stage detection pipeline on an uploaded image.

    Parameters
    ----------
    enforce_liveness:
        Set ``True`` **only** for live camera captures.  When ``False``
        (default) the liveness check is skipped entirely so that uploaded
        photos are processed without triggering false SPOOF_BLOCKED results.
    tenant_id:
        Tenant scope for filtering matches and recording audit entries.

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
            await _add_spoof_audit(session, query_hash, gps_lat=gps_lat, gps_lon=gps_lon, tenant_id=tenant_id)
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
    matches = await verify_match(embedding, session, limit=limit, tenant_id=tenant_id)

    if matches:
        return {
            "status": "MATCH",
            "matches": matches,
            "query_hash": query_hash,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "tenant_id": tenant_id,
        }
    return {
        "status": "NO_MATCH",
        "matches": [],
        "query_hash": query_hash,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "tenant_id": tenant_id,
    }


# ---------- Registration Pipeline ----------
async def register_pipeline(
    file: UploadFile,
    session,
    face_name: str,
    alias: Optional[str] = None,
    demographics: Optional[dict] = None,
    tenant_id: int = 1,
) -> dict:
    """
    Register a new face: stages 1–4 + database insert.

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

    profile_id = await register_profile(session, face_name, alias, demographics, embedding, tenant_id=tenant_id)

    query_hash = compute_query_hash(embedding.tobytes())
    await add_audit_entry(
        session,
        event_type="REGISTER",
        query_hash=query_hash,
        result_name=face_name,
        tenant_id=tenant_id,
    )

    return {
        "status": "REGISTERED",
        "profile_id": profile_id,
        "query_hash": query_hash,
        "embedding_dim": int(embedding.shape[0]),
        "tenant_id": tenant_id,
    }


# ---------- Re-enrolment Pipeline ----------
async def reenroll_pipeline(
    file: UploadFile,
    session,
    face_id: int,
    face_name: str,
    tenant_id: int = 1,
) -> dict:
    """
    Re-enrol an existing face: stages 1–4 then UPDATE the stored embedding.

    No liveness check is performed (same rationale as register_pipeline —
    re-enrolment always uses uploaded mugshots / ID photos, not live captures).

    Returns a dict with keys:
      status        — "RE_ENROLLED" | "ERROR"
      profile_id    — the face's existing DB id
      query_hash    — SHA-256 of the *new* embedding bytes
      embedding_dim — 512 for ArcFace
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

    from app.db.vector_ops import add_audit_entry, update_face_embedding

    updated = await update_face_embedding(session, face_id, embedding, tenant_id=tenant_id)
    if not updated:
        return {"status": "ERROR", "error": f"Face {face_id} not found in database"}

    query_hash = compute_query_hash(embedding.tobytes())
    await add_audit_entry(
        session,
        event_type="REGISTER_REENROLL",
        query_hash=query_hash,
        result_name=face_name,   # no raw PII, just name tag for audit trail
        tenant_id=tenant_id,
    )

    return {
        "status": "RE_ENROLLED",
        "profile_id": face_id,
        "query_hash": query_hash,
        "embedding_dim": int(embedding.shape[0]),
        "tenant_id": tenant_id,
    }
