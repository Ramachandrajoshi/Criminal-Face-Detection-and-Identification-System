"""
Search endpoint — upload query image → run pipeline → return match or NO_MATCH.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.pipeline import run_pipeline
from app.core.config import settings
from app.db.session import get_session
from app.db.vector_ops import add_audit_entry, compute_query_hash
from app.schemas.face import MatchResult, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_face(
    file: UploadFile = File(..., description="Face image to search (JPEG/PNG, ≤ 5 MB)"),
    search_data: Optional[SearchRequest] = None,
    limit: int = Query(10, ge=1, le=50, description="Max matches to return"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Search for a matching suspect profile against the database.
    
    Returns the best match if distance ≤ MATCH_THRESHOLD, otherwise NO_MATCH.
    All searches are logged to the audit trail.
    """
    # Validate file
    if not file.content_type or file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be ≤ 5 MB")

    # Rewind file so run_pipeline can read it
    file.file.seek(0)

    gps_lat = search_data.gps_lat if search_data else None
    gps_lon = search_data.gps_lon if search_data else None

    result = await run_pipeline(file, session, gps_lat, gps_lon)

    # Log to audit trail
    query_hash = result.get("query_hash", compute_query_hash(b""))
    event_type = result["status"]

    result_name = None
    if result["status"] == "MATCH" and result.get("matches"):
        result_name = result["matches"][0]["suspect_name"]

    await add_audit_entry(
        session,
        event_type=event_type,
        query_hash=query_hash,
        result_name=result_name,
        distance=result["matches"][0]["distance"] if result["matches"] else None,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )

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
    )
