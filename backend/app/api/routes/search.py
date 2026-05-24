"""
Search endpoint — upload query image → run pipeline → return match or NO_MATCH.

Liveness policy
---------------
Liveness checking is enabled **only** when the caller passes
``is_live_capture=true`` (i.e. the frame came from a live webcam/camera feed).
Photo uploads must pass ``is_live_capture=false`` (the default) so that static
images are not incorrectly blocked with SPOOF_BLOCKED.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.pipeline import run_pipeline
from app.core.validation import validate_image_dimensions
from app.db.session import get_session
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
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Search for a matching suspect profile against the database.

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
    )

    # ── Audit trail ──────────────────────────────────────────────
    query_hash = result.get("query_hash", compute_query_hash(b""))
    event_type = result["status"]

    result_name: Optional[str] = None
    if event_type == "MATCH" and result.get("matches"):
        result_name = result["matches"][0]["suspect_name"]

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
        )

    if event_type == "MATCH" and audit_id is not None:
        best = result["matches"][0] if result.get("matches") else None
        alert_id = await create_alert(
            session,
            audit_log_id=audit_id,
            suspect_id=best.get("id") if best else None,
            event_type="MATCH",
            distance=best.get("distance") if best else None,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
        )

    # ── Response ─────────────────────────────────────────────────
    matches = [
        MatchResult(
            id=m["id"],
            suspect_name=m["suspect_name"],
            alias=m["alias"],
            distance=m["distance"],
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
