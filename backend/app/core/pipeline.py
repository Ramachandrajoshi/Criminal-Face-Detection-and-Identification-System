"""
Core pipeline — five-stage face processing pipeline.

Stage 1 — DETECT   : OpenCV frame grab → RetinaFace bounding box
Stage 2 — ALIGN    : Eye-landmark affine transform
Stage 3 — NORMALIZE: Resize & pixel normalisation
Stage 4 — REPRESENT: deepface.represent() → 512-d ArcFace embedding
Stage 5 — VERIFY   : pgvector cosine ANN query

Embedding versions 1–3 run stages 1–4 through DeepFace (``detect_face`` +
``extract_embedding``). Version 4 runs them through the InsightFace ONNX
engine (``embed_with_insightface``, see app/core/face_engine.py). Stage 5 is
shared: each version is queried against its own vectors.

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

import asyncio
import logging
from typing import Optional

import numpy as np
from fastapi import UploadFile

from app.core.config import (
    CURRENT_EMBEDDING_VERSION,
    DEEPFACE_EMBEDDING_VERSIONS,
    INSIGHTFACE_EMBEDDING_VERSION,
    SUPPORTED_EMBEDDING_VERSIONS,
    match_threshold_for_version,
    settings,
)
from app.core.debug_dump import dump_detection, dump_note, dump_stage, face_dump
from app.core.inference_executor import run_inference
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

    attempts: list[dict] = []
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
                area = best.get("facial_area", {})
                x, y = area.get("x", 0), area.get("y", 0)
                eyes = [area[k] for k in ("left_eye", "right_eye") if area.get(k)]
                attempts.append({"backend": backend, "faces": len(faces), "chosen": True})
                dump_note("deepface_detect_attempts", attempts)
                dump_detection(
                    "dfc_01_detect", frame,
                    (x, y, x + area.get("w", 0), y + area.get("h", 0)),
                    landmarks=eyes or None,
                    label=f"{backend} {best.get('confidence', 0):.2f}",
                    backend=backend, confidence=best.get("confidence"),
                )
                dump_stage("dfc_02_crop", best["face"], rgb=True, backend=backend)
                return best["face"]

            logger.debug(
                "Backend %s found %d face(s) but all below confidence threshold %.2f — trying next",
                backend, len(faces), _MIN_FACE_CONFIDENCE,
            )
            attempts.append({
                "backend": backend,
                "faces": len(faces),
                "confidences": [f.get("confidence") for f in faces],
                "rejected": f"below {_MIN_FACE_CONFIDENCE}",
            })

        except ValueError:
            # DeepFace raises ValueError("Face could not be detected ...") when
            # no face bounding box is found with enforce_detection=True.
            # This is the expected path for images without faces — not an error.
            logger.debug("No face bounding box found by detector '%s' — trying next", backend)
            attempts.append({"backend": backend, "faces": 0})
        except Exception as exc:
            # Real driver / model failures (e.g. OOM, missing weights).
            logger.warning("Detector '%s' errored: %s — trying next backend", backend, exc)
            attempts.append({"backend": backend, "error": str(exc)})

    logger.warning(
        "No face detected in image after trying all detector backends %s. "
        "Image will be marked as FAILED.",
        [settings.deepface_detector, "mtcnn", "opencv"],
    )
    dump_note("deepface_detect_attempts", attempts)
    return None


# ---------- Stage 2-3: ALIGN + NORMALIZE ----------

def _preprocess_for_arcface(
    face: np.ndarray, version: int = CURRENT_EMBEDDING_VERSION
) -> np.ndarray:
    """
    Prepare an aligned face array for ArcFace embedding extraction.

    Returns RGB uint8. Version 3 fixes square padding; older recipes retain
    their geometry for compatibility with already-enrolled embeddings.

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

    if version not in DEEPFACE_EMBEDDING_VERSIONS:
        raise ValueError(
            f"Unsupported embedding version: {version} "
            f"(supported: {DEEPFACE_EMBEDDING_VERSIONS})"
        )

    # ── Step 0: uint8 conversion ─────────────────────────────────────────────
    if face.dtype != np.uint8:
        face = (face * 255.0).clip(0, 255).astype(np.uint8)

    # At this point face is RGB uint8.
    # Convert to BGR once for OpenCV processing; we'll convert back at the end.
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    # Debug dump names: v<recipe>_<step>_<name>; a gap means the step was off.
    tag = f"v{version}"
    dump_stage(f"{tag}_03_uint8", face_bgr)

    # ── Step 1: Upscale tiny crops ───────────────────────────────────────────
    # CCTV face crops can be as small as 50×50 px. Upscaling to at least
    # `small_crop_min_side` before enhancement gives the subsequent filters
    # significantly more pixels to work with and reduces blocking artefacts.
    if version >= 2 and settings.enable_upscale_small_crops:
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
            dump_stage(f"{tag}_04_upscaled", face_bgr, scale=round(scale, 3))

    # ── Step 2: Denoising ────────────────────────────────────────────────────
    # Non-local means denoising removes CCTV sensor/compression noise before
    # the contrast equaliser runs. h=5 is conservative — strong enough to
    # smooth noise without blurring eye or lip edges.
    if version >= 2 and settings.enable_denoising:
        face_bgr = cv2.fastNlMeansDenoisingColored(
            face_bgr,
            None,
            h=5,
            hColor=5,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        dump_stage(f"{tag}_05_denoised", face_bgr)

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
    if version >= 2 and settings.enable_gamma_correction:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        if 1.0 < mean_lum < 255.0:  # avoid log(0) and division by log(1)
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
            dump_stage(f"{tag}_06_gamma", face_bgr, input_mean_lum=round(mean_lum, 1), gamma=round(gamma, 3))


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
        dump_stage(f"{tag}_07_clahe", face_bgr, clip_limit=settings.clahe_clip_limit)

    # ── Step 5: Unsharp mask ─────────────────────────────────────────────────
    if version >= 2 and settings.enable_unsharp_mask:
        blurred = cv2.GaussianBlur(face_bgr, (0, 0), sigmaX=2.0)
        s = float(settings.unsharp_strength)
        face_bgr = cv2.addWeighted(face_bgr, s, blurred, -(s - 1.0), 0)
        dump_stage(f"{tag}_08_unsharp", face_bgr, strength=s)

    # ── Step 6: Pad → 112×112 Lanczos resize ─────────────────────────────────
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    if face.shape[:2] != (112, 112):
        h, w = face.shape[:2]
        side = max(h, w)
        if version >= 3:
            top = (side - h) // 2
            bottom = side - h - top
            left = (side - w) // 2
            right = side - w - left
        else:
            # Preserve legacy geometry for queries against existing vectors.
            top, bottom = divmod(side - h, 2)
            left, right = divmod(side - w, 2)
        face = cv2.copyMakeBorder(
            face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LANCZOS4)

    dump_stage(f"{tag}_09_final_112", face, rgb=True)
    return face


# ---------- Stage 4: REPRESENT ----------
def extract_embedding(
    aligned_face: np.ndarray, version: int = CURRENT_EMBEDDING_VERSION
) -> Optional[np.ndarray]:
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
    try:
        from deepface import DeepFace

        processed_face = _preprocess_for_arcface(aligned_face, version=version)
        if version >= 3:
            # extract_faces returns RGB; represent's ndarray API expects BGR,
            # including detector_backend="skip". Legacy vectors used RGB here.
            processed_face = np.ascontiguousarray(processed_face[:, :, ::-1])

        result = DeepFace.represent(
            img_path=processed_face,
            model_name=settings.deepface_model,
            detector_backend="skip",       # already aligned + preprocessed
            enforce_detection=False,
            normalization=settings.arcface_normalization,
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


# ---------- Stages 1-4 for version 4: InsightFace ----------
def embed_with_insightface(image_bytes: bytes, enrollment: bool = False) -> Optional[np.ndarray]:
    """
    Detect, align and embed the primary face with the InsightFace engine.

    ``enrollment`` picks the centred face (mugshots, ID photos); otherwise the
    largest fully visible face is used, wherever it is (see
    ``select_primary_face``).

    Returns the L2-normalised 512-d embedding, or None when no face is found.
    Engine failures (missing weights, ONNX Runtime errors) raise — callers
    treat those as an embedding failure rather than "no face".
    """
    import cv2

    from app.core.face_engine import get_engine

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("Failed to decode image bytes")
        return None

    engine = get_engine()
    best, route = engine.detect_primary(frame, prefer_center=enrollment)
    if best is None:
        logger.debug("InsightFace: no face above det threshold")
        dump_note("insightface_detect", "no face above det threshold")
        return None

    logger.debug("InsightFace: face score=%.2f area=%.0f pass=%s", best.score, best.area, route)
    dump_detection(
        "v4_01_detect", frame, best.bbox, best.landmarks,
        label=f"scrfd {best.score:.2f}", score=best.score, detect_pass=route,
    )
    aligned = engine.align(frame, best.landmarks)
    dump_stage("v4_02_aligned_112", aligned)
    return engine.embed(aligned)


class _NoFace:
    """Marker: the engine ran and found no face (as opposed to failing)."""


_NO_FACE = _NoFace()


async def _embed_versions(content: bytes, versions: list[int], enrollment: bool = False) -> dict:
    """
    Embed ``content`` under each requested version.

    Returns {version: embedding | None | _NO_FACE}. DeepFace versions share one
    detection pass; InsightFace runs its own detector. Both engines run
    concurrently on the inference pool. ``enrollment`` only changes which face
    InsightFace picks when there are several.
    """
    deepface_versions = [v for v in versions if v in DEEPFACE_EMBEDDING_VERSIONS]

    async def run_deepface() -> dict:
        if not deepface_versions:
            return {}
        aligned = await run_inference(detect_face, content)
        if aligned is None:
            return {v: _NO_FACE for v in deepface_versions}
        # Independent work — run it on the inference pool concurrently rather
        # than serially, so wall-clock stays at roughly one embedding.
        embeddings = await asyncio.gather(
            *(run_inference(extract_embedding, aligned, v) for v in deepface_versions)
        )
        return dict(zip(deepface_versions, embeddings))

    async def run_insightface() -> dict:
        if INSIGHTFACE_EMBEDDING_VERSION not in versions:
            return {}
        try:
            embedding = await run_inference(embed_with_insightface, content, enrollment)
        except Exception as exc:  # noqa: BLE001 — engine failure, not "no face"
            logger.error("InsightFace embedding failed: %s", exc)
            return {INSIGHTFACE_EMBEDDING_VERSION: None}
        return {INSIGHTFACE_EMBEDDING_VERSION: _NO_FACE if embedding is None else embedding}

    deepface_result, insightface_result = await asyncio.gather(run_deepface(), run_insightface())
    return {**deepface_result, **insightface_result}


async def _embed_for_enrollment(content: bytes) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Embed under the current recipe. Returns (embedding, error message)."""
    result = (await _embed_versions(content, [CURRENT_EMBEDDING_VERSION], enrollment=True))[CURRENT_EMBEDDING_VERSION]
    if result is _NO_FACE:
        return None, "No face detected in the uploaded image"
    if result is None:
        return None, "Embedding extraction failed"
    return result, None


# ---------- Debug face dump ----------
def _dump_matches(matches: list[dict]) -> list[dict]:
    """Match summary for the debug manifest — ids and distances, no names."""
    return [
        {"face_id": m["id"], "embedding_version": m.get("embedding_version"), "distance": m["distance"]}
        for m in matches
    ]


async def _dump_nearest(session, embeddings: dict, tenant_id: int) -> None:
    """
    Debug dump only: record the 5 nearest vectors per recipe *ignoring* the
    match threshold, so a NO_MATCH still shows how far the right person was.
    """
    from app.db.vector_ops import cosine_ann_query

    nearest = {}
    for v, emb in embeddings.items():
        if emb is None:
            continue
        rows = await cosine_ann_query(
            session, emb.tolist(), threshold=2.0, limit=5, tenant_id=tenant_id, target_version=v
        )
        nearest[f"v{v}"] = {"threshold": match_threshold_for_version(v), "nearest": _dump_matches(rows)}
    dump_note("nearest_ignoring_threshold", nearest)


# ---------- Stage 5: VERIFY (ANN query) ----------
async def verify_match(
    embedding: np.ndarray,
    session,
    threshold: Optional[float] = None,
    limit: int = 10,
    tenant_id: Optional[int] = None,
    target_version: Optional[int] = None,
) -> list[dict]:
    """
    Execute pgvector cosine ANN query with the given embedding.
    Returns list of match dicts sorted by ascending distance.
    """
    from app.db.vector_ops import cosine_ann_query

    if threshold is None:
        threshold = match_threshold_for_version(
            target_version if target_version is not None else CURRENT_EMBEDDING_VERSION
        )
    query_hash = compute_query_hash(embedding.tobytes())
    matches = await cosine_ann_query(
        session,
        embedding.tolist(),
        threshold=threshold,
        limit=limit,
        tenant_id=tenant_id if tenant_id is not None else 1,
        target_version=target_version,
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

    Embedding recipes
    -----------------
    A stored vector can only be compared against a query embedded with the
    same recipe, so the query is embedded once per recipe that actually has
    rows for this tenant (see ``get_present_embedding_versions``) — after a
    version bump that is typically a single recipe, not all of them. Those
    embeddings are independent and run concurrently on the inference pool.

    Every recipe's hits are then merged and ranked by distance together, so a
    closer match under an older recipe is never masked by a weaker one under a
    newer recipe. ``limit`` bounds the merged set.

    The current recipe is always embedded, even when nothing is enrolled under
    it yet, so ``query_hash`` identifies the query image rather than whichever
    recipe happened to hold the match — audit entries for the same image stay
    comparable across a re-enrolment.

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
    with face_dump("search", content) as dump:
        result = await _search(content, session, gps_lat, gps_lon, limit, enforce_liveness, tenant_id)
        if dump is not None:
            dump.note("result", {
                "status": result["status"],
                "error": result.get("error"),
                "matches": _dump_matches(result.get("matches", [])),
            })
    return result


async def _search(
    content: bytes,
    session,
    gps_lat: Optional[float],
    gps_lon: Optional[float],
    limit: int,
    enforce_liveness: bool,
    tenant_id: int,
) -> dict:
    """Body of ``run_pipeline`` once the upload has been read."""
    # ── Liveness gate (camera only) ───────────────────────────────
    if enforce_liveness:
        is_live = await run_inference(check_liveness_on_bytes, content)
        if not is_live:
            query_hash = compute_query_hash(b"")
            await _add_spoof_audit(session, query_hash, gps_lat=gps_lat, gps_lon=gps_lon, tenant_id=tenant_id)
            return {
                "status": "SPOOF_BLOCKED",
                "matches": [],
                "query_hash": query_hash,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
            }

    # ── Stages 1–4: Detect → Represent ────────────────────────────
    # A stored vector is only comparable to a query embedded with the same
    # recipe, so the query is embedded once per recipe still in use. Recipes
    # with no enrolled rows for this tenant can never match, so they are not
    # embedded at all — after a version bump that is every recipe but one.
    from app.db.vector_ops import get_present_embedding_versions

    present = await get_present_embedding_versions(session, tenant_id=tenant_id)
    searchable = [v for v in SUPPORTED_EMBEDDING_VERSIONS if v in present]

    # The current recipe is always embedded, even with nothing enrolled under
    # it yet: it is what gives the audit trail a hash that depends only on the
    # query image, not on which recipe happened to hold the match.
    to_embed = sorted({*searchable, CURRENT_EMBEDDING_VERSION}, reverse=True)

    results = await _embed_versions(content, to_embed)
    if all(r is _NO_FACE for r in results.values()):
        return {"status": "ERROR", "error": "No face detected", "matches": []}

    # An engine that found no face where another did is treated like a failed
    # extraction for its recipes: those vectors were not compared.
    embeddings = {v: (None if r is _NO_FACE else r) for v, r in results.items()}
    if settings.debug_face_dump:
        await _dump_nearest(session, {v: embeddings[v] for v in searchable}, tenant_id)

    # Prefer the current recipe's hash; fall back to the newest that succeeded
    # so a failed extraction still leaves a usable audit trail.
    query_hash = next(
        (
            compute_query_hash(embeddings[v].tobytes())
            for v in (CURRENT_EMBEDDING_VERSION, *to_embed)
            if embeddings.get(v) is not None
        ),
        compute_query_hash(b""),
    )

    # ── Stage 5: Verify ───────────────────────────────────────────
    # Each recipe is queried against its own vectors, then the results are
    # merged and ranked by distance. Ranking globally rather than returning
    # the newest recipe's hits means a closer match in an older recipe is
    # never masked by a weaker one in a newer recipe.
    #
    # Different models have different distance scales, so hits are ranked by
    # distance relative to their own recipe's threshold. Recipes that share a
    # threshold (1–3) therefore still rank purely by distance.
    searched = [v for v in searchable if embeddings[v] is not None]
    per_version = await asyncio.gather(
        *(
            verify_match(
                embeddings[v], session, threshold=match_threshold_for_version(v),
                limit=limit, tenant_id=tenant_id, target_version=v,
            )
            for v in searched
        )
    )

    ranked = sorted(
        ((m["distance"] / match_threshold_for_version(v), m)
         for v, group in zip(searched, per_version) for m in group),
        key=lambda pair: pair[0],
    )
    matches = [m for _, m in ranked[:limit]]

    if matches:
        return {
            "status": "MATCH",
            "matches": matches,
            "query_hash": query_hash,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "tenant_id": tenant_id,
        }

    if len(searched) != len(searchable):
        # A recipe holding enrolled vectors could not be embedded, so those
        # vectors were never compared — that is not a complete NO_MATCH.
        # A failure of the current recipe while nothing is enrolled under it
        # costs only hash stability and does not reach here.
        return {
            "status": "ERROR", "error": "Embedding extraction failed", "matches": [],
            "query_hash": query_hash,
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

    from app.db.vector_ops import add_audit_entry, register_profile

    with face_dump("register", content):
        # Stages 1–4 under the current recipe (EMBEDDING_VERSION)
        embedding, error = await _embed_for_enrollment(content)
        if embedding is None:
            dump_note("result", {"status": "ERROR", "error": error})
            return {"status": "ERROR", "error": error}

        profile_id = await register_profile(session, face_name, alias, demographics, embedding, tenant_id=tenant_id)
        dump_note("result", {"status": "REGISTERED", "face_id": profile_id})

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

    # Stages 1–4 under the current recipe (EMBEDDING_VERSION)
    with face_dump(f"reenroll_face{face_id}", content):
        embedding, error = await _embed_for_enrollment(content)
        dump_note("result", {"status": "ERROR" if embedding is None else "OK", "error": error})
    if embedding is None:
        return {"status": "ERROR", "error": error}

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
