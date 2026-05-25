"""
Register endpoint — upload image + metadata → extract & store embedding.
Supports single registration, batch registration, and a streaming batch endpoint
that pushes Server-Sent Events (SSE) with per-file progress for real-time UI.

Session strategy for the SSE endpoint
--------------------------------------
FastAPI's `Depends(get_session)` yields a **single** AsyncSession that is bound
to one asyncpg connection for the lifetime of the HTTP request.  For the SSE
streaming endpoint that connection would be held open for the full batch
duration (potentially minutes).  asyncpg / PostgreSQL idle-session timeouts will
kill that connection mid-batch and subsequent `GET /alerts` requests that happen
to receive the same pooled connection get ``connection is closed``.

Fix: the streaming generator acquires a **fresh session per file** directly from
`async_session_factory`.  Each session lives only for the duration of one
pipeline call (a few hundred ms) and is cleanly closed and returned to the pool
immediately afterwards.  The DI-injected session is intentionally not used inside
`generate()`.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.pipeline import register_pipeline
from app.core.validation import validate_image_dimensions
from app.db.models import SuspectProfile as SuspectProfileModel
from app.db.session import async_session_factory, get_session
from app.db.vector_ops import add_audit_entry, compute_query_hash
from app.schemas.face import RegisterResponse, SuspectProfileOut, SuspectUpdateIn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["register"])

ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}
MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


# ── Helpers ──────────────────────────────────────────────────────


def _name_from_filename(filename: str) -> str:
    """
    Derive a human-readable suspect name from an uploaded filename.

    Rules:
    1. Strip the extension.
    2. Replace underscores and hyphens with spaces.
    3. Title-case the result.
    4. Remove leading/trailing whitespace.

    Examples:
      "john_doe.jpg"       → "John Doe"
      "Jane-Smith-02.png"  → "Jane Smith 02"
      "suspect123.jpeg"    → "Suspect123"
    """
    stem = Path(filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned.title() if cleaned else "Unknown"


async def _validate_file(file: UploadFile) -> tuple[bytes, str]:
    """
    Validate MIME type, file size, and image dimensions.
    Returns (raw_bytes, error_message).  error_message is '' on success.
    """
    if not file.content_type or file.content_type not in ALLOWED_MIME:
        return b"", f"Unsupported MIME type: {file.content_type!r}"

    content = await file.read()

    if len(content) > MAX_SIZE_BYTES:
        return b"", "Image exceeds 5 MB limit"

    try:
        validate_image_dimensions(content)
    except ValueError as exc:
        return b"", str(exc)

    return content, ""


def _sse_event(data: dict) -> str:
    """Format a dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(data)}\n\n"


# ── Single registration ──────────────────────────────────────────


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register_suspect(
    file: UploadFile = File(..., description="Face image (JPEG/PNG, ≤ 5 MB)"),
    suspect_name: Optional[str] = Form(
        None,
        max_length=100,
        description="Suspect full name (auto-derived from filename if omitted)",
    ),
    alias: Optional[str] = Form(None, max_length=100, description="Known alias"),
    demographics: Optional[str] = Form(
        None, description="JSON demographics (age_band, gender, ethnicity)"
    ),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Register a new suspect profile.

    - ``suspect_name`` is optional — derived from the filename when omitted.
    - All other metadata fields are optional.
    - The system extracts a 512-d ArcFace embedding and stores it encrypted.
    """
    content, err = await _validate_file(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    resolved_name = (suspect_name or "").strip() or _name_from_filename(file.filename or "unknown")

    demographics_dict: Optional[dict] = None
    if demographics:
        try:
            demographics_dict = json.loads(demographics)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid demographics JSON")

    file.file.seek(0)
    result = await register_pipeline(file, session, resolved_name, alias, demographics_dict)

    if result["status"] in ("ERROR", "SPOOF_BLOCKED"):
        raise HTTPException(status_code=422, detail=result.get("error", result["status"]))

    return RegisterResponse(
        status=result["status"],
        profile_id=result.get("profile_id"),
        query_hash=result["query_hash"],
        embedding_dim=result.get("embedding_dim"),
    )


# ── Shared helper: register one file with a dedicated session ─────


async def _register_one(
    file: UploadFile,
    suspect_name: str,
    alias: Optional[str],
    demographics_dict: Optional[dict],
) -> dict:
    """
    Register a single file using its own short-lived AsyncSession.

    Key design decisions
    --------------------
    * A fresh session is opened per file so long-running SSE batches never
      hold a connection across the full batch duration.
    * **This function never raises.**  It always returns a result dict with
      ``status`` set to ``"ERROR"`` on failure.  This is critical: if an
      exception propagated out of the ``async with`` block, SQLAlchemy would
      call ``do_terminate()`` (an async operation) while inside an anyio
      cancel scope that Starlette may already be tearing down — producing
      ``asyncio.CancelledError`` spam in the logs.  By catching all
      exceptions and returning an error dict instead, the ``async with``
      context always exits cleanly.
    """
    async with async_session_factory() as session:
        try:
            return await register_pipeline(
                file, session, suspect_name, alias, demographics_dict
            )
        except Exception as exc:
            logger.exception("register_pipeline raised for %s", suspect_name)
            # Best-effort rollback — ignore secondary errors.
            try:
                await session.rollback()
            except Exception:  # pragma: no cover
                pass
            # Return an error dict instead of re-raising so that the
            # ``async with`` context exits without an active exception,
            # keeping the asyncpg connection cleanup path clean.
            return {"status": "ERROR", "error": str(exc)}


# ── Batch registration (JSON response, no streaming) ─────────────


@router.post("/register/batch", status_code=201)
async def register_suspects_batch(
    files: List[UploadFile] = File(
        ..., description="One or more face images (JPEG/PNG, each ≤ 5 MB)"
    ),
    alias: Optional[str] = Form(None, max_length=100),
    demographics: Optional[str] = Form(None),
    _user: dict = Depends(get_current_user),
):
    """
    Batch-register multiple suspects (waits for all files, returns JSON).
    For real-time progress streaming, use ``POST /register/batch/stream``.

    Note: no DI session is injected here — each file gets its own session
    via ``_register_one`` to avoid holding a connection open for the full
    batch duration.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Batch limit is 50 images per request")

    demographics_dict: Optional[dict] = None
    if demographics:
        try:
            demographics_dict = json.loads(demographics)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid demographics JSON")

    results: list[dict] = []

    for file in files:
        filename = file.filename or "unknown"
        suspect_name = _name_from_filename(filename)

        content, err = await _validate_file(file)
        if err:
            results.append({
                "filename": filename, "status": "ERROR",
                "profileId": None, "suspectName": suspect_name, "error": err,
            })
            continue

        file.file.seek(0)
        try:
            result = await _register_one(file, suspect_name, alias, demographics_dict)
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected error registering %s", filename)
            results.append({
                "filename": filename, "status": "ERROR",
                "profileId": None, "suspectName": suspect_name, "error": str(exc),
            })
            continue

        if result["status"] in ("ERROR", "SPOOF_BLOCKED"):
            results.append({
                "filename": filename, "status": result["status"],
                "profileId": None, "suspectName": suspect_name, "error": result.get("error"),
            })
        else:
            results.append({
                "filename": filename, "status": "REGISTERED",
                "profileId": result.get("profile_id"), "suspectName": suspect_name, "error": None,
            })

    registered = sum(1 for r in results if r["status"] == "REGISTERED")
    return {
        "totalFiles": len(results),
        "registered": registered,
        "failed": len(results) - registered,
        "results": results,
    }


# ── Streaming batch registration (SSE) ───────────────────────────


@router.post("/register/batch/stream")
async def register_suspects_batch_stream(
    files: List[UploadFile] = File(
        ..., description="One or more face images (JPEG/PNG, each ≤ 5 MB)"
    ),
    alias: Optional[str] = Form(None, max_length=100),
    demographics: Optional[str] = Form(None),
    _user: dict = Depends(get_current_user),
):
    """
    Streaming batch-register using Server-Sent Events (SSE).

    Processes files sequentially and emits one ``data:`` event per file the
    moment it completes.  The client reads these events to render a live
    progress bar with ETA.

    **Important**: no ``session`` is injected via DI here.  Each file
    registration uses its own short-lived session from ``_register_one()``.
    This prevents a single asyncpg connection being held open for the full
    batch, which would cause ``connection is closed`` errors on subsequent
    requests once the server-side idle timeout fires.

    **SSE event payload schema** (JSON):
    ```json
    {
      "type":        "start" | "progress" | "done",
      "processed":   1,
      "total":       5,
      "filename":    "john.jpg",
      "suspectName": "John Doe",
      "status":      "REGISTERED" | "ERROR" | "SPOOF_BLOCKED",
      "profileId":   42,
      "error":       null,
      "elapsedMs":   1234,
      "fileMs":      380
    }
    ```
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Batch limit is 50 images per request")

    demographics_dict: Optional[dict] = None
    if demographics:
        try:
            demographics_dict = json.loads(demographics)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid demographics JSON")

    total = len(files)

    async def generate() -> AsyncGenerator[str, None]:
        # Send the opening frame immediately so the browser's SSE connection
        # is established before the first (heavy) pipeline call.
        yield _sse_event({"type": "start", "total": total, "processed": 0, "elapsedMs": 0})
        await asyncio.sleep(0)   # flush headers

        batch_start = time.perf_counter()
        processed = 0
        registered = 0
        failed = 0

        for file in files:
            filename = file.filename or "unknown"
            suspect_name = _name_from_filename(filename)
            file_start = time.perf_counter()

            # ── Validate ─────────────────────────────────────────
            content, err = await _validate_file(file)
            if err:
                processed += 1
                failed += 1
                yield _sse_event({
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "filename": filename,
                    "suspectName": suspect_name,
                    "status": "ERROR",
                    "profileId": None,
                    "error": err,
                    "elapsedMs": int((time.perf_counter() - batch_start) * 1000),
                    "fileMs": int((time.perf_counter() - file_start) * 1000),
                })
                await asyncio.sleep(0)
                continue

            # ── Pipeline (own session per file) ──────────────────
            file.file.seek(0)
            try:
                result = await _register_one(file, suspect_name, alias, demographics_dict)
            except Exception as exc:  # pragma: no cover
                logger.exception("Unexpected error registering %s", filename)
                processed += 1
                failed += 1
                yield _sse_event({
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "filename": filename,
                    "suspectName": suspect_name,
                    "status": "ERROR",
                    "profileId": None,
                    "error": str(exc),
                    "elapsedMs": int((time.perf_counter() - batch_start) * 1000),
                    "fileMs": int((time.perf_counter() - file_start) * 1000),
                })
                await asyncio.sleep(0)
                continue

            file_ms = int((time.perf_counter() - file_start) * 1000)
            elapsed_ms = int((time.perf_counter() - batch_start) * 1000)
            processed += 1

            if result["status"] in ("ERROR", "SPOOF_BLOCKED"):
                failed += 1
                yield _sse_event({
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "filename": filename,
                    "suspectName": suspect_name,
                    "status": result["status"],
                    "profileId": None,
                    "error": result.get("error"),
                    "elapsedMs": elapsed_ms,
                    "fileMs": file_ms,
                })
            else:
                registered += 1
                yield _sse_event({
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "filename": filename,
                    "suspectName": suspect_name,
                    "status": "REGISTERED",
                    "profileId": result.get("profile_id"),
                    "error": None,
                    "elapsedMs": elapsed_ms,
                    "fileMs": file_ms,
                })

            await asyncio.sleep(0)   # yield to event loop between files

        # ── Final summary event ───────────────────────────────────
        total_ms = int((time.perf_counter() - batch_start) * 1000)
        yield _sse_event({
            "type": "done",
            "processed": processed,
            "total": total,
            "registered": registered,
            "failed": failed,
            "totalMs": total_ms,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            "Connection": "keep-alive",
        },
    )


# ── Suspect CRUD ─────────────────────────────────────────────────


@router.get("/suspects", response_model=list[SuspectProfileOut])
async def list_suspects(
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    List all registered suspect profiles.
    Embeddings are NEVER returned — metadata only.
    """
    result = await session.execute(
        select(
            SuspectProfileModel.id,
            SuspectProfileModel.suspect_name,
            SuspectProfileModel.alias,
            SuspectProfileModel.demographics,
            SuspectProfileModel.created_at,
        ).order_by(SuspectProfileModel.id.desc())
    )
    rows = result.all()
    return [
        SuspectProfileOut(
            id=r.id,
            suspect_name=r.suspect_name,
            alias=r.alias,
            demographics=r.demographics,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]


@router.get("/suspects/{suspect_id}", response_model=SuspectProfileOut)
async def get_suspect(
    suspect_id: int,
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """Get a single suspect profile by ID (metadata only)."""
    result = await session.execute(
        select(
            SuspectProfileModel.id,
            SuspectProfileModel.suspect_name,
            SuspectProfileModel.alias,
            SuspectProfileModel.demographics,
            SuspectProfileModel.created_at,
        ).where(SuspectProfileModel.id == suspect_id)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Suspect not found")
    return SuspectProfileOut(
        id=row.id,
        suspect_name=row.suspect_name,
        alias=row.alias,
        demographics=row.demographics,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


@router.patch("/suspects/{suspect_id}", response_model=SuspectProfileOut)
async def update_suspect(
    suspect_id: int,
    body: SuspectUpdateIn,
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Update a suspect's metadata (name / alias / demographics).
    The face embedding is never modified via this endpoint.
    """
    result = await session.execute(
        select(SuspectProfileModel).where(SuspectProfileModel.id == suspect_id)
    )
    suspect = result.scalar_one_or_none()
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found")

    if body.suspect_name is not None:
        suspect.suspect_name = body.suspect_name.strip()
    if body.alias is not None:
        suspect.alias = body.alias.strip() or None
    if body.demographics is not None:
        suspect.demographics = body.demographics

    await session.commit()
    await session.refresh(suspect)

    return SuspectProfileOut(
        id=suspect.id,
        suspect_name=suspect.suspect_name,
        alias=suspect.alias,
        demographics=suspect.demographics,
        created_at=suspect.created_at.isoformat() if suspect.created_at else "",
    )


@router.delete("/suspects/{suspect_id}", status_code=204)
async def delete_suspect(
    suspect_id: int,
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Hard-delete a suspect profile.
    Logs a REGISTER_DELETE audit entry BEFORE deleting (audit log is append-only).
    Alerts referencing this suspect are left untouched (operator must dismiss manually).
    """
    result = await session.execute(
        select(
            SuspectProfileModel.id,
            SuspectProfileModel.suspect_name,
        ).where(SuspectProfileModel.id == suspect_id)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Suspect not found")

    # Write audit entry BEFORE delete (audit log is append-only)
    await add_audit_entry(
        session,
        event_type="REGISTER_DELETE",
        query_hash=compute_query_hash(str(suspect_id).encode()),
        result_name=None,   # no PII in logs — only suspect_id via query_hash
        distance=None,
        gps_lat=None,
        gps_lon=None,
    )

    await session.execute(
        sa_delete(SuspectProfileModel).where(SuspectProfileModel.id == suspect_id)
    )
    await session.commit()
    # 204 No Content
