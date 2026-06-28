"""
Audit log endpoint — read-only audit trail (admin token required).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.db.session import get_session
from app.schemas.face import AuditEntryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["audit"])


@router.get("/audit", response_model=list[AuditEntryResponse])
async def get_audit_log(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    tenant_id: int = Query(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Retrieve audit log entries (read-only, append-only table).
    Admin access only — requires valid JWT.
    """
    if _user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    offset = (page - 1) * page_size
    params = {"limit": page_size, "offset": offset, "tenant_id": tenant_id}

    where_clause = ""
    if event_type:
        where_clause = "AND event_type = :event_type"
        params["event_type"] = event_type

    sql = text(
        f"""
        SELECT
            id, event_type, query_hash, result_name,
            distance, gps_lat, gps_lon, timestamp, tenant_id
        FROM audit_log
        WHERE tenant_id = :tenant_id
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT :limit OFFSET :offset
        """
    )

    result = await session.execute(sql, params)
    rows = result.fetchall()

    entries = []
    for row in rows:
        entries.append(AuditEntryResponse(
            id=row[0],
            event_type=row[1],
            query_hash=row[2],
            result_name=row[3],
            distance=row[4],
            gps_lat=row[5],
            gps_lon=row[6],
            timestamp=row[7].isoformat() if row[7] else None,
            tenant_id=row[8],
        ))

    return entries
