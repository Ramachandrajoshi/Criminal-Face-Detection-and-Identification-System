"""
Pose-aware enrollment and search — for CCTV frames where the face is turned
sideways, up or down rather than looking straight at the camera.

Endpoints
---------
POST   /api/v1/pose/register                      new profile from 1–10 photos at different angles
POST   /api/v1/pose/faces/{face_id}/poses         add angle photos to an existing profile (oldest dropped past the limit)
PUT    /api/v1/pose/faces/{face_id}/poses         re-register: replace all of a profile's poses
GET    /api/v1/pose/faces/{face_id}/poses         list a profile's enrolled poses
DELETE /api/v1/pose/faces/{face_id}/poses/{pid}   remove one pose
POST   /api/v1/pose/search                        match a frame against every enrolled pose
POST   /api/v1/pose/estimate                      head pose of a frame only (webcam guidance)

These are additive: the existing /register, /search and /faces endpoints are
unchanged. New poses are embedded under POSE_EMBEDDING_VERSION: version 5
(YOLOv5-Face + AdaFace) when EMBEDDING_VERSION=5, otherwise version 4
(InsightFace). Search embeds the query once per version that has poses for
the tenant, so poses enrolled under the other version keep matching until the
face is re-registered.

How matching works
------------------
Each photo is tagged with an estimated head pose (yaw / pitch / roll from the
detector's 5 landmarks) and stored as its own embedding. Search compares the
query with all of them in a single SQL statement per version and keeps each
person's closest pose. Frames turned beyond POSE_MAX_YAW_DEG / POSE_MAX_PITCH_DEG
return LOW_QUALITY instead of a guess.
"""

import asyncio
import json
import logging
from typing import List, Optional

import numpy as np

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.register import _validate_file
from app.core.auth import get_current_user
from app.core.config import (
    POSE_EMBEDDING_VERSION,
    POSE_EMBEDDING_VERSIONS,
    match_threshold_for_version,
    settings,
)
from app.core.debug_dump import dump_note, face_dump
from app.core.inference_executor import run_inference
from app.core.pose_pipeline import PoseEmbedding, embed_with_pose, estimate_frame_pose
from app.db.models import FaceProfile as FaceProfileModel
from app.db.pose_ops import (
    add_poses,
    delete_pose,
    get_present_pose_versions,
    list_poses,
    pose_ann_query,
    register_profile_with_poses,
    replace_poses,
)
from app.db.session import get_session
from app.db.vector_ops import add_audit_entry, compute_query_hash, create_alert
from app.schemas.pose import (
    PoseEstimateResponse,
    PoseImageResult,
    PoseInfo,
    PoseMatchResult,
    PoseOut,
    PoseRegisterResponse,
    PoseSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pose", tags=["pose"])

MAX_FILES_PER_REQUEST = 10


# ── Helpers ──────────────────────────────────────────────────────


def _pose_info(result: PoseEmbedding) -> PoseInfo:
    return PoseInfo(
        yaw=round(result.pose.yaw, 1),
        pitch=round(result.pose.pitch, 1),
        roll=round(result.pose.roll, 1),
        bucket=result.pose.bucket,
        quality=round(result.quality, 3),
    )


async def _embed_upload(file: UploadFile) -> tuple[Optional[PoseEmbedding], Optional[str]]:
    """Validate one upload and embed it. Returns (result, error); never raises."""
    content, err = await _validate_file(file)
    if err:
        return None, err
    # One debug dump folder per uploaded angle (no-op unless DEBUG_FACE_DUMP).
    with face_dump("pose_enroll", content):
        try:
            result = await run_inference(embed_with_pose, content, True, POSE_EMBEDDING_VERSION)
        except Exception as exc:  # noqa: BLE001 — engine failure, reported per file
            logger.error("Pose embedding failed for %s: %s", file.filename, exc)
            dump_note("result", {"status": "ERROR", "error": str(exc)})
            return None, "Embedding extraction failed"
    if result is None:
        return None, "No face detected in the uploaded image"
    if not result.accepted:
        return result, result.rejected_reason
    return result, None


async def _embed_uploads(files: List[UploadFile]) -> list[tuple[str, Optional[PoseEmbedding], Optional[str]]]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400, detail=f"Limit is {MAX_FILES_PER_REQUEST} images per request"
        )
    # The inference pool bounds real concurrency; gather keeps input order.
    outcomes = await asyncio.gather(*(_embed_upload(f) for f in files))
    return [(f.filename or "unknown", r, e) for f, (r, e) in zip(files, outcomes)]


def _image_results(
    embedded: list[tuple[str, Optional[PoseEmbedding], Optional[str]]],
    pose_ids: list[int],
) -> list[PoseImageResult]:
    """Per-file report; ``pose_ids`` are in the order the accepted files appear."""
    ids = iter(pose_ids)
    return [
        PoseImageResult(
            filename=name,
            status="REJECTED" if err else "ACCEPTED",
            pose_id=None if err else next(ids),
            pose=_pose_info(result) if result else None,
            error=err,
        )
        for name, result, err in embedded
    ]


def _accepted(embedded) -> list[PoseEmbedding]:
    return [r for _, r, err in embedded if err is None]


def _template(accepted: list[PoseEmbedding]) -> np.ndarray:
    """Normalised average of the accepted photos: the profile's main embedding."""
    template = np.mean([r.embedding for r in accepted], axis=0).astype(np.float32)
    return template / np.linalg.norm(template)


def _rejection_detail(embedded) -> dict:
    return {
        "error": "No usable face in any uploaded image",
        "images": [
            {"filename": name, "error": err} for name, _, err in embedded
        ],
    }


# ── Registration ─────────────────────────────────────────────────


@router.post("/register", response_model=PoseRegisterResponse, status_code=201)
async def register_with_poses(
    files: List[UploadFile] = File(
        ...,
        description=(
            "1–10 photos of the same person at different angles: frontal, "
            "turned left/right, looking up/down (JPEG/PNG, each ≤ 5 MB)"
        ),
    ),
    face_name: str = Form(..., min_length=1, max_length=100),
    alias: Optional[str] = Form(None, max_length=100),
    demographics: Optional[str] = Form(None, description="JSON demographics"),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Register a new face profile from several photos at different head poses.

    Every usable photo is stored as its own pose-tagged embedding. Their
    average becomes the profile's main embedding (POSE_EMBEDDING_VERSION), so the person is
    found by the standard ``/search`` endpoint too. An average of several
    photos matches far better than any single one: on LFW, three averaged
    photos found 99.1% of low-resolution probes at the configured threshold,
    one photo 96.6%.

    Photos with no face, or turned beyond the pose limits, are reported as
    ``REJECTED``; the profile is created as long as at least one is usable.
    """
    demographics_dict: Optional[dict] = None
    if demographics:
        try:
            demographics_dict = json.loads(demographics)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid demographics JSON")

    embedded = await _embed_uploads(files)
    accepted = _accepted(embedded)
    if not accepted:
        raise HTTPException(status_code=422, detail=_rejection_detail(embedded))
    if len(accepted) > settings.pose_max_per_face:
        raise HTTPException(
            status_code=400,
            detail=f"A face can hold at most {settings.pose_max_per_face} poses",
        )

    template = _template(accepted)
    profile_id, pose_ids = await register_profile_with_poses(
        session,
        face_name.strip(),
        alias,
        demographics_dict,
        template,
        [(r.embedding, r.pose, r.quality) for r in accepted],
        tenant_id=tenant_id,
        embedding_version=POSE_EMBEDDING_VERSION,
    )

    await add_audit_entry(
        session,
        event_type="REGISTER_POSES",
        query_hash=compute_query_hash(template.tobytes()),
        result_name=face_name.strip(),
        tenant_id=tenant_id,
    )

    return PoseRegisterResponse(
        status="REGISTERED",
        profile_id=profile_id,
        accepted=len(accepted),
        rejected=len(embedded) - len(accepted),
        pose_buckets=sorted({r.pose.bucket for r in accepted}),
        images=_image_results(embedded, pose_ids),
        tenant_id=tenant_id,
    )


async def _require_face(session: AsyncSession, face_id: int, tenant_id: int) -> str:
    row = (
        await session.execute(
            select(FaceProfileModel.face_name).where(
                FaceProfileModel.id == face_id, FaceProfileModel.tenant_id == tenant_id
            )
        )
    ).one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Face not found")
    return row.face_name


@router.post("/faces/{face_id}/poses", response_model=PoseRegisterResponse, status_code=201)
async def add_face_poses(
    face_id: int,
    files: List[UploadFile] = File(..., description="1–10 photos at different angles"),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Add pose photos to an existing profile — including profiles registered
    through the standard ``/register`` endpoint. The profile's main embedding
    is not changed, so the standard ``/search`` behaves exactly as before.

    A profile keeps its newest POSE_MAX_PER_FACE poses: once it is full, each
    new photo replaces the oldest one, so re-registering through this endpoint
    never runs into the limit. ``PUT`` replaces all poses at once instead.
    """
    face_name = await _require_face(session, face_id, tenant_id)

    embedded = await _embed_uploads(files)
    accepted = _accepted(embedded)
    if not accepted:
        raise HTTPException(status_code=422, detail=_rejection_detail(embedded))
    if len(accepted) > settings.pose_max_per_face:
        raise HTTPException(
            status_code=400,
            detail=f"A face can hold at most {settings.pose_max_per_face} poses",
        )

    pose_ids, dropped = await add_poses(
        session,
        face_id,
        [(r.embedding, r.pose, r.quality) for r in accepted],
        tenant_id=tenant_id,
        max_poses=settings.pose_max_per_face,
        embedding_version=POSE_EMBEDDING_VERSION,
    )

    await add_audit_entry(
        session,
        event_type="REGISTER_POSES",
        query_hash=compute_query_hash(accepted[0].embedding.tobytes()),
        result_name=face_name,
        tenant_id=tenant_id,
    )
    if dropped:
        await add_audit_entry(
            session,
            event_type="REGISTER_POSES_DROP_OLDEST",
            query_hash=compute_query_hash(f"{face_id}:{','.join(map(str, dropped))}".encode()),
            tenant_id=tenant_id,
        )

    return PoseRegisterResponse(
        status="POSES_ADDED",
        profile_id=face_id,
        accepted=len(accepted),
        rejected=len(embedded) - len(accepted),
        pose_buckets=sorted({r.pose.bucket for r in accepted}),
        images=_image_results(embedded, pose_ids),
        tenant_id=tenant_id,
    )


@router.put("/faces/{face_id}/poses", response_model=PoseRegisterResponse)
async def replace_face_poses(
    face_id: int,
    files: List[UploadFile] = File(..., description="1–10 photos at different angles"),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Re-register a profile: replace all of its poses with these photos.

    The old pose embeddings are deleted and the profile's main embedding
    becomes the average of the new photos (POSE_EMBEDDING_VERSION), as with
    ``/pose/register``, so the old photos no longer match in ``/search`` or
    ``/pose/search``. Name, alias and demographics are kept. When no photo is
    usable the request fails with 422 and the profile is left unchanged.
    """
    face_name = await _require_face(session, face_id, tenant_id)

    embedded = await _embed_uploads(files)
    accepted = _accepted(embedded)
    if not accepted:
        raise HTTPException(status_code=422, detail=_rejection_detail(embedded))
    if len(accepted) > settings.pose_max_per_face:
        raise HTTPException(
            status_code=400,
            detail=f"A face can hold at most {settings.pose_max_per_face} poses",
        )

    template = _template(accepted)
    pose_ids = await replace_poses(
        session,
        face_id,
        template,
        [(r.embedding, r.pose, r.quality) for r in accepted],
        tenant_id=tenant_id,
        embedding_version=POSE_EMBEDDING_VERSION,
    )
    if pose_ids is None:  # deleted while the photos were being embedded
        raise HTTPException(status_code=404, detail="Face not found")

    await add_audit_entry(
        session,
        event_type="REGISTER_POSES_REPLACE",
        query_hash=compute_query_hash(template.tobytes()),
        result_name=face_name,
        tenant_id=tenant_id,
    )

    return PoseRegisterResponse(
        status="POSES_REPLACED",
        profile_id=face_id,
        accepted=len(accepted),
        rejected=len(embedded) - len(accepted),
        pose_buckets=sorted({r.pose.bucket for r in accepted}),
        images=_image_results(embedded, pose_ids),
        tenant_id=tenant_id,
    )


@router.get("/faces/{face_id}/poses", response_model=list[PoseOut])
async def get_face_poses(
    face_id: int,
    tenant_id: int = Query(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """List a profile's enrolled poses (metadata only, never embeddings)."""
    await _require_face(session, face_id, tenant_id)
    return [PoseOut(**p) for p in await list_poses(session, face_id, tenant_id=tenant_id)]


@router.delete("/faces/{face_id}/poses/{pose_id}", status_code=204)
async def remove_face_pose(
    face_id: int,
    pose_id: int,
    tenant_id: int = Query(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """Delete one pose embedding. The profile and its other poses are kept."""
    if not await delete_pose(session, face_id, pose_id, tenant_id=tenant_id):
        raise HTTPException(status_code=404, detail="Pose not found")
    await add_audit_entry(
        session,
        event_type="REGISTER_POSE_DELETE",
        query_hash=compute_query_hash(f"{face_id}:{pose_id}".encode()),
        tenant_id=tenant_id,
    )


# ── Live guidance ────────────────────────────────────────────────


@router.post("/estimate", response_model=PoseEstimateResponse)
async def estimate_pose_of_frame(
    file: UploadFile = File(..., description="Webcam frame (JPEG/PNG, ≤ 5 MB)"),
    _user: dict = Depends(get_current_user),
):
    """
    Measure the head pose in one frame — face detection only.

    Used by the webcam enrollment screen to guide the subject ("turn left…
    hold") several times a second. Nothing is embedded, stored, matched or
    audited: the response says where the face is and how it is turned.
    """
    content, err = await _validate_file(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    estimate = await run_inference(estimate_frame_pose, content, POSE_EMBEDDING_VERSION)
    if estimate is None:
        return PoseEstimateResponse(face_found=False, reason="No face detected")

    pose = estimate.pose
    return PoseEstimateResponse(
        face_found=True,
        pose=PoseInfo(
            yaw=round(pose.yaw, 1),
            pitch=round(pose.pitch, 1),
            roll=round(pose.roll, 1),
            bucket=pose.bucket,
            quality=round(estimate.quality, 3),
        ),
        bbox=[round(v, 4) for v in estimate.bbox],
        usable=estimate.rejected_reason is None,
        reason=estimate.rejected_reason,
    )


# ── Search ───────────────────────────────────────────────────────


async def _embed_query(content: bytes, version: int) -> tuple[Optional[PoseEmbedding], bool]:
    """
    Embed a search frame under one version, whatever its pose (the caller
    decides). Returns (result, failed): result None means no face, or an
    engine failure when ``failed``.
    """
    try:
        return await run_inference(embed_with_pose, content, False, version, False), False
    except Exception as exc:  # noqa: BLE001 — engine failure, reported as ERROR
        logger.error("Pose embedding (version %d) failed: %s", version, exc)
        return None, True


def _merge_matches(per_version: dict[int, list[dict]], limit: int) -> list[dict]:
    """
    One entry per person across versions, ranked by distance relative to its
    version's threshold (as ``/search`` ranks across versions). A person with
    poses of both versions keeps the better match.
    """
    best: dict[int, tuple[float, dict]] = {}
    for version, group in per_version.items():
        threshold = match_threshold_for_version(version)
        for match in group:
            score = match["distance"] / threshold
            if match["id"] not in best or score < best[match["id"]][0]:
                best[match["id"]] = (score, match)
    return [m for _, m in sorted(best.values(), key=lambda pair: pair[0])][:limit]


@router.post("/search", response_model=PoseSearchResponse)
async def search_with_poses(
    file: UploadFile = File(..., description="CCTV frame or photo (JPEG/PNG, ≤ 5 MB)"),
    gps_lat: Optional[float] = Form(None, ge=-90, le=90),
    gps_lon: Optional[float] = Form(None, ge=-180, le=180),
    limit: int = Query(10, ge=1, le=50, description="Max people to return"),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Match a face at any reasonable angle against every enrolled pose.

    Each person appears at most once, scored by their pose closest to the
    query. ``queryPose`` reports the measured head pose; ``matchedPose`` on
    each match says which enrolled angle matched.

    Statuses: ``MATCH`` | ``NO_MATCH`` | ``LOW_QUALITY`` (face turned beyond
    the pose limits — not searched) | ``ERROR``. Every search is audited and a
    MATCH raises an alert, as with ``/search``.

    The frame is embedded under POSE_EMBEDDING_VERSION and under every other
    version that has poses for the tenant; each is compared only with poses of
    its own version. The query pose, the pose limits and the query hash come
    from POSE_EMBEDDING_VERSION (another version's when it finds no face), and
    ``matchThreshold`` is that version's threshold.
    """
    content, err = await _validate_file(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    threshold = match_threshold_for_version(POSE_EMBEDDING_VERSION)
    query_pose: Optional[PoseInfo] = None
    matches: list[dict] = []
    query_hash = compute_query_hash(b"")

    with face_dump("pose_search", content):
        present = {v for v in await get_present_pose_versions(session, tenant_id=tenant_id)
                   if v in POSE_EMBEDDING_VERSIONS}
        # The configured version first: it decides the pose and the hash.
        versions = [POSE_EMBEDDING_VERSION, *sorted(present - {POSE_EMBEDDING_VERSION}, reverse=True)]
        outcomes = dict(zip(versions, await asyncio.gather(*(_embed_query(content, v) for v in versions))))
        found = {v: result for v, (result, _) in outcomes.items() if result is not None}
        lead = next((found[v] for v in versions if v in found), None)

        if lead is None:
            engine_failed = any(f for _, f in outcomes.values())
            status, error = "ERROR", "Embedding extraction failed" if engine_failed else "No face detected"
        elif not lead.accepted:
            status, error = "LOW_QUALITY", lead.rejected_reason
        else:
            status, error = None, None

        if lead is not None:
            query_pose = _pose_info(lead)

        if status is None:
            query_hash = compute_query_hash(lead.embedding.tobytes())
            # Sequential: one AsyncSession must not run statements concurrently.
            per_version = {
                v: await pose_ann_query(
                    session, found[v].embedding, threshold=match_threshold_for_version(v),
                    limit=limit, tenant_id=tenant_id, embedding_version=v,
                )
                for v in versions if v in found
            }
            matches = _merge_matches(per_version, limit)
            if matches:
                status = "MATCH"
            elif present - found.keys():
                # Poses of a version that found no face were never compared.
                status, error = "ERROR", "Embedding extraction failed"
            else:
                status = "NO_MATCH"
        dump_note("result", {
            "status": status, "error": error, "threshold": threshold,
            "versions": {v: ("compared" if v in found else "no face or failed") for v in versions},
            "matches": [{"face_id": m["id"], "distance": m["distance"]} for m in matches],
        })
        if settings.debug_face_dump and lead is not None and lead.accepted:
            # Nearest profiles ignoring the threshold, to see how far a miss was.
            nearest = [
                {"version": v, "face_id": m["id"], "distance": m["distance"]}
                for v in found
                for m in await pose_ann_query(
                    session, found[v].embedding, threshold=2.0, limit=5,
                    tenant_id=tenant_id, embedding_version=v,
                )
            ]
            dump_note("nearest_ignoring_threshold", nearest)

    best = matches[0] if matches else None
    audit_id = await add_audit_entry(
        session,
        event_type=status,
        query_hash=query_hash,
        result_name=best["face_name"] if best else None,
        distance=best["distance"] if best else None,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        tenant_id=tenant_id,
    )

    alert_id: Optional[int] = None
    if best:
        alert_id = await create_alert(
            session,
            audit_log_id=audit_id,
            face_id=best["id"],
            event_type="MATCH",
            distance=best["distance"],
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            tenant_id=tenant_id,
        )

    return PoseSearchResponse(
        status=status,
        query_hash=query_hash,
        query_pose=query_pose,
        matches=[PoseMatchResult(**m) for m in matches],
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        match_threshold=threshold,
        alert_id=alert_id,
        error=error,
    )
