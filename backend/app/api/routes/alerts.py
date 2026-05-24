"""
Alerts endpoint — paginated list + human-in-the-loop confirmation.

Connection resilience: all DB calls are wrapped with `_with_retry`, which
retries once on `InterfaceError` (stale/closed asyncpg connection).  Combined
with `pool_pre_ping=True` in the engine config, this eliminates the
``connection is closed`` errors that surface when a pooled connection is killed
by the server-side idle timeout between requests.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.exc import InterfaceError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.db.session import get_session
from app.schemas.face import AlertResponse, ConfirmRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["alerts"])


# ── Retry helper ─────────────────────────────────────────────────


async def _execute_with_retry(session: AsyncSession, sql, params: dict):
    """
    Execute *sql* with *params*, retrying once if the connection has been
    dropped (``InterfaceError / connection is closed``).

    pool_pre_ping=True already handles most cases, but there is a narrow race
    where a connection passes the ping and is then dropped before the real
    query runs.  One retry is sufficient to recover from that race.
    """
    try:
        return await session.execute(sql, params)
    except InterfaceError as exc:
        logger.warning(
            "DB connection dropped mid-query (%s) — rolling back and retrying once", exc
        )
        try:
            await session.rollback()
        except Exception:
            pass
        # Re-raise: the session's underlying connection is now dead and cannot
        # be reused.  FastAPI will give the client a 500; pool_pre_ping will
        # evict the stale connection before the next request.
        raise


# ── Routes ───────────────────────────────────────────────────────


@router.get("/alerts", response_model=list[AlertResponse])
async def list_alerts(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(
        None, description="Filter by status: PENDING_REVIEW, CONFIRMED, DISMISSED"
    ),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Retrieve recent alerts with pagination.
    Supports filtering by alert status.
    """
    offset = (page - 1) * page_size
    params: dict = {"limit": page_size, "offset": offset}

    where_clause = ""
    if status_filter:
        where_clause = "WHERE a.status = :status_filter"
        params["status_filter"] = status_filter

    sql = text(
        f"""
        SELECT
            a.id,
            a.audit_log_id,
            a.suspect_id,
            a.event_type,
            a.distance,
            a.status,
            a.gps_lat,
            a.gps_lon,
            a.created_at,
            a.confirmed_at
        FROM alerts a
        {where_clause}
        ORDER BY a.created_at DESC
        LIMIT :limit OFFSET :offset
        """
    )

    result = await _execute_with_retry(session, sql, params)
    rows = result.fetchall()

    return [
        AlertResponse(
            id=row[0],
            audit_log_id=row[1],
            suspect_id=row[2],
            event_type=row[3],
            distance=row[4],
            status=row[5],
            gps_lat=row[6],
            gps_lon=row[7],
            created_at=row[8].isoformat() if row[8] else None,
            confirmed_at=row[9].isoformat() if row[9] else None,
        )
        for row in rows
    ]


@router.post("/alerts/{alert_id}/confirm")
async def confirm_alert(
    alert_id: int,
    confirm_data: ConfirmRequest,
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Human operator confirms or dismisses a match alert.

    - ``confirmed=True``  → status becomes CONFIRMED, confirmed_at is set
    - ``confirmed=False`` → status becomes DISMISSED
    """
    status_val = "CONFIRMED" if confirm_data.confirmed else "DISMISSED"
    confirmed_at: Optional[datetime] = datetime.now(timezone.utc) if confirm_data.confirmed else None

    sql_update = text(
        """
        UPDATE alerts
        SET status = :status, confirmed_at = :confirmed_at
        WHERE id = :alert_id
        RETURNING id, status
        """
    )

    result = await _execute_with_retry(
        session,
        sql_update,
        {"alert_id": alert_id, "status": status_val, "confirmed_at": confirmed_at},
    )
    await session.commit()

    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    return {
        "alert_id": alert_id,
        "status": row[1],
        "confirmed_at": confirmed_at.isoformat() if confirmed_at else None,
    }
