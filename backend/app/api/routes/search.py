"""
Search endpoint — upload query image → run pipeline → return match or NO_MATCH.

Liveness policy
---------------
Liveness checking is enabled **only** when the caller passes
``is_live_capture=true`` (i.e. the frame came from a live webcam/camera feed).
Photo uploads must pass ``is_live_capture=false`` (the default) so that static
images are not incorrectly blocked with SPOOF_BLOCKED.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.pipeline import run_pipeline
from app.core.validation import validate_image_dimensions
from app.db.session import async_session_factory, get_session
from app.db.vector_ops import add_audit_entry, compute_query_hash, create_alert
from app.schemas.face import MatchResult, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_face(
    file: UploadFile = File(..., description="Face image to search (JPEG/PNG, ≤ 5 MB)"),
    gps_lat: Optional[float] = Form(None, ge=-90, le=90, description="GPS latitude of capture"),
    gps_lon: Optional[float] = Form(None, ge=-180, le=180, description="GPS longitude of capture"),
    is_live_capture: bool = Form(
        False,
        description=(
            "Set true when the image comes from a live camera feed. "
            "Enables liveness / anti-spoofing check.  "
            "Must be false (default) for uploaded photo files."
        ),
    ),
    limit: int = Query(10, ge=1, le=50, description="Max matches to return"),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Search for a matching face profile against the database.

    - **Photo uploads**: set ``is_live_capture=false`` (default) — liveness check skipped.
    - **Live camera frames**: set ``is_live_capture=true`` — liveness / anti-spoofing enforced.

    Returns the best match if cosine distance ≤ MATCH_THRESHOLD, otherwise NO_MATCH.
    All searches (including SPOOF_BLOCKED) are logged to the audit trail.
    """
    # ── Input validation ─────────────────────────────────────────
    if not file.content_type or file.content_type not in (
        "image/jpeg", "image/png", "image/jpg"
    ):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be ≤ 5 MB")

    try:
        validate_image_dimensions(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Rewind so run_pipeline can read it
    file.file.seek(0)

    # ── Run pipeline ─────────────────────────────────────────────
    result = await run_pipeline(
        file,
        session,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        limit=limit,
        enforce_liveness=is_live_capture,   # only live camera frames trigger liveness
        tenant_id=tenant_id,
    )

    # ── Audit trail ──────────────────────────────────────────────
    query_hash = result.get("query_hash", compute_query_hash(b""))
    event_type = result["status"]

    result_name: Optional[str] = None
    if event_type == "MATCH" and result.get("matches"):
        result_name = result["matches"][0]["face_name"]

    audit_id: Optional[int] = None
    alert_id: Optional[int] = None

    # SPOOF_BLOCKED audit is written inside run_pipeline; don't double-log
    if event_type != "SPOOF_BLOCKED":
        audit_id = await add_audit_entry(
            session,
            event_type=event_type,
            query_hash=query_hash,
            result_name=result_name,
            distance=result["matches"][0]["distance"] if result.get("matches") else None,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            tenant_id=tenant_id,
        )

    if event_type == "MATCH" and audit_id is not None:
        best = result["matches"][0] if result.get("matches") else None
        alert_id = await create_alert(
            session,
            audit_log_id=audit_id,
            face_id=best.get("id") if best else None,
            event_type="MATCH",
            distance=best.get("distance") if best else None,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            tenant_id=tenant_id,
        )

    # ── Response ─────────────────────────────────────────────────
    matches = [
        MatchResult(
            id=m["id"],
            face_name=m["face_name"],
            alias=m["alias"],
            distance=m["distance"],
            embedding_version=m.get("embedding_version", 1),
            tenant_id=m.get("tenant_id", tenant_id),
        )
        for m in result.get("matches", [])
    ]

    return SearchResponse(
        status=result["status"],
        query_hash=query_hash,
        matches=matches,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        match_threshold=settings.match_threshold,
        alert_id=alert_id,
    )


async def _search_one(
    file: UploadFile,
    gps_lat: Optional[float],
    gps_lon: Optional[float],
    limit: int,
    tenant_id: int = 1,
) -> dict:
    """
    Run the full search pipeline for a single file using its own short-lived session.
    Never raises — returns an error dict on failure.
    """
    async with async_session_factory() as session:
        try:
            result = await run_pipeline(
                file,
                session,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                limit=limit,
                enforce_liveness=False,  # batch photo upload — never live capture
                tenant_id=tenant_id,
            )

            query_hash = result.get("query_hash", compute_query_hash(b""))
            event_type = result["status"]

            result_name: Optional[str] = None
            if event_type == "MATCH" and result.get("matches"):
                result_name = result["matches"][0]["face_name"]

            audit_id: Optional[int] = None
            alert_id: Optional[int] = None

            if event_type != "SPOOF_BLOCKED":
                audit_id = await add_audit_entry(
                    session,
                    event_type=event_type,
                    query_hash=query_hash,
                    result_name=result_name,
                    distance=result["matches"][0]["distance"] if result.get("matches") else None,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    tenant_id=tenant_id,
                )

            if event_type == "MATCH" and audit_id is not None:
                best = result["matches"][0] if result.get("matches") else None
                alert_id = await create_alert(
                    session,
                    audit_log_id=audit_id,
                    face_id=best.get("id") if best else None,
                    event_type="MATCH",
                    distance=best.get("distance") if best else None,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    tenant_id=tenant_id,
                )

            return {
                "status": event_type,
                "query_hash": query_hash,
                "matches": result.get("matches", []),
                "alert_id": alert_id,
            }
        except Exception as exc:
            logger.exception("batch search pipeline failed")
            try:
                await session.rollback()
            except Exception:
                pass
            return {"status": "ERROR", "error": str(exc), "query_hash": "", "matches": [], "alert_id": None}


def _search_sse_event(data: dict) -> str:
    """Format a dict as a single SSE data frame."""
    import json
    return f"data: {json.dumps(data)}\n\n"


async def _process_search_file(
    file: UploadFile,
    gps_lat: Optional[float],
    gps_lon: Optional[float],
    limit: int,
    tenant_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Validate + search a single batch file, bounded by ``semaphore``.
    Never raises — always returns a dict describing the outcome.
    """
    filename = file.filename or "unknown"
    file_start = time.perf_counter()

    async with semaphore:
        if not file.content_type or file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            return {
                "filename": filename, "status": "ERROR", "queryHash": "", "matches": [],
                "alertId": None, "fileMs": int((time.perf_counter() - file_start) * 1000),
                "error": f"Unsupported MIME type: {file.content_type}",
            }

        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            return {
                "filename": filename, "status": "ERROR", "queryHash": "", "matches": [],
                "alertId": None, "fileMs": int((time.perf_counter() - file_start) * 1000),
                "error": "Image exceeds 5 MB limit",
            }

        try:
            validate_image_dimensions(content)
        except ValueError as exc:
            return {
                "filename": filename, "status": "ERROR", "queryHash": "", "matches": [],
                "alertId": None, "fileMs": int((time.perf_counter() - file_start) * 1000),
                "error": str(exc),
            }

        file.file.seek(0)
        result = await _search_one(file, gps_lat, gps_lon, limit, tenant_id=tenant_id)

        matches_out = [
            {
                "id": m["id"],
                "suspectName": m.get("face_name", m.get("person_name")),
                "alias": m.get("alias"),
                "distance": m["distance"],
                "embeddingVersion": m.get("embedding_version", 1),
                "tenantId": m.get("tenant_id", tenant_id),
            }
            for m in result.get("matches", [])
        ]

        return {
            "filename": filename,
            "status": result["status"],
            "queryHash": result.get("query_hash", ""),
            "matches": matches_out,
            "alertId": result.get("alert_id"),
            "fileMs": int((time.perf_counter() - file_start) * 1000),
            "error": result.get("error"),
        }


@router.post("/search/batch/stream")
async def search_faces_batch_stream(
    files: List[UploadFile] = File(
        ..., description="1–20 face images (JPEG/PNG, ≤ 5 MB each)"
    ),
    gps_lat: Optional[float] = Form(None, ge=-90, le=90),
    gps_lon: Optional[float] = Form(None, ge=-180, le=180),
    limit: int = Query(10, ge=1, le=50),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    _user: dict = Depends(get_current_user),
):
    """
    Batch face search using Server-Sent Events (SSE).

    Processes up to 20 images concurrently (bounded by
    ``settings.batch_pipeline_concurrency``) through the full 5-stage
    pipeline. Emits one SSE progress event per file the moment it completes —
    **completion order is not the same as submission order** since files run
    concurrently; each event carries its own ``filename`` so clients should
    key off that rather than assuming order.
    Each MATCH creates an alert (status PENDING_REVIEW) and an audit log entry.
    is_live_capture is always False for batch photo uploads.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Batch limit is 20 images per request")

    total = len(files)

    async def generate() -> AsyncGenerator[str, None]:
        yield _search_sse_event({"type": "start", "total": total, "processed": 0, "elapsedMs": 0})
        await asyncio.sleep(0)

        batch_start = time.perf_counter()
        processed = 0
        matched = 0
        no_match = 0
        errors = 0

        semaphore = asyncio.Semaphore(settings.batch_pipeline_concurrency)
        tasks = [
            asyncio.create_task(_process_search_file(f, gps_lat, gps_lon, limit, tenant_id, semaphore))
            for f in files
        ]

        for coro in asyncio.as_completed(tasks):
            file_result = await coro
            processed += 1

            status = file_result["status"]
            if status == "MATCH":
                matched += 1
            elif status == "NO_MATCH":
                no_match += 1
            else:
                errors += 1

            yield _search_sse_event({
                "type": "progress",
                "processed": processed,
                "total": total,
                "filename": file_result["filename"],
                "status": status,
                "queryHash": file_result["queryHash"],
                "matches": file_result["matches"],
                "alertId": file_result["alertId"],
                "elapsedMs": int((time.perf_counter() - batch_start) * 1000),
                "fileMs": file_result["fileMs"],
                "error": file_result["error"],
            })
            await asyncio.sleep(0)

        total_ms = int((time.perf_counter() - batch_start) * 1000)
        yield _search_sse_event({
            "type": "done",
            "processed": processed,
            "total": total,
            "matched": matched,
            "noMatch": no_match,
            "errors": errors,
            "totalMs": total_ms,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
