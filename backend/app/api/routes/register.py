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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.pipeline import reenroll_pipeline, register_pipeline
from app.core.validation import validate_image_dimensions
from app.db.models import FaceProfile as FaceProfileModel
from app.db.session import async_session_factory, get_session
from app.db.vector_ops import add_audit_entry, compute_query_hash
from app.schemas.face import ReenrollResponse, RegisterResponse, SuspectProfileOut, SuspectUpdateIn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["register"])

ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}
MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


# ── Helpers ──────────────────────────────────────────────────────


def _name_from_filename(filename: str) -> str:
    """
    Derive a human-readable face name from an uploaded filename.

    Rules:
    1. Strip the extension.
    2. Replace underscores and hyphens with spaces.
    3. Title-case the result.
    4. Remove leading/trailing whitespace.

    Examples:
      "john_doe.jpg"       → "John Doe"
      "Jane-Smith-02.png"  → "Jane Smith 02"
      "face123.jpeg"       → "Face123"
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
async def register_face(
    file: UploadFile = File(..., description="Face image (JPEG/PNG, ≤ 5 MB)"),
    face_name: Optional[str] = Form(
        None,
        max_length=100,
        description="Face full name (auto-derived from filename if omitted)",
    ),
    alias: Optional[str] = Form(None, max_length=100, description="Known alias"),
    demographics: Optional[str] = Form(
        None, description="JSON demographics (age_band, gender, ethnicity)"
    ),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Register a new face profile.

    - ``face_name`` is optional — derived from the filename when omitted.
    - All other metadata fields are optional.
    - The system extracts a 512-d ArcFace embedding and stores it encrypted.
    """
    content, err = await _validate_file(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    resolved_name = (face_name or "").strip() or _name_from_filename(file.filename or "unknown")

    demographics_dict: Optional[dict] = None
    if demographics:
        try:
            demographics_dict = json.loads(demographics)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid demographics JSON")

    file.file.seek(0)
    result = await register_pipeline(file, session, resolved_name, alias, demographics_dict, tenant_id=tenant_id)

    if result["status"] in ("ERROR", "SPOOF_BLOCKED"):
        raise HTTPException(status_code=422, detail=result.get("error", result["status"]))

    return RegisterResponse(
        status=result["status"],
        profile_id=result.get("profile_id"),
        query_hash=result["query_hash"],
        embedding_dim=result.get("embedding_dim"),
        tenant_id=result.get("tenant_id", tenant_id),
    )


# ── Shared helper: register one file with a dedicated session ─────


async def _register_one(
    file: UploadFile,
    face_name: str,
    alias: Optional[str],
    demographics_dict: Optional[dict],
    tenant_id: int = 1,
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
                file, session, face_name, alias, demographics_dict, tenant_id=tenant_id
            )
        except Exception as exc:
            logger.exception("register_pipeline raised for %s", face_name)
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
async def register_faces_batch(
    files: List[UploadFile] = File(
        ..., description="One or more face images (JPEG/PNG, each ≤ 5 MB)"
    ),
    alias: Optional[str] = Form(None, max_length=100),
    demographics: Optional[str] = Form(None),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    _user: dict = Depends(get_current_user),
):
    """
    Batch-register multiple faces (waits for all files, returns JSON).
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
        face_name = _name_from_filename(filename)

        content, err = await _validate_file(file)
        if err:
            results.append({
                "filename": filename, "status": "ERROR",
                "profileId": None, "faceName": face_name, "error": err,
            })
            continue

        file.file.seek(0)
        try:
            result = await _register_one(file, face_name, alias, demographics_dict, tenant_id=tenant_id)
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected error registering %s", filename)
            results.append({
                "filename": filename, "status": "ERROR",
                "profileId": None, "faceName": face_name, "error": str(exc),
            })
            continue

        if result["status"] in ("ERROR", "SPOOF_BLOCKED"):
            results.append({
                "filename": filename, "status": result["status"],
                "profileId": None, "faceName": face_name, "error": result.get("error"),
            })
        else:
            results.append({
                "filename": filename, "status": "REGISTERED",
                "profileId": result.get("profile_id"), "faceName": face_name, "error": None,
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
async def register_faces_batch_stream(
    files: List[UploadFile] = File(
        ..., description="One or more face images (JPEG/PNG, each ≤ 5 MB)"
    ),
    alias: Optional[str] = Form(None, max_length=100),
    demographics: Optional[str] = Form(None),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
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
      "faceName":    "John Doe",
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
            face_name = _name_from_filename(filename)
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
                    "faceName": face_name,
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
                result = await _register_one(file, face_name, alias, demographics_dict, tenant_id=tenant_id)
            except Exception as exc:  # pragma: no cover
                logger.exception("Unexpected error registering %s", filename)
                processed += 1
                failed += 1
                yield _sse_event({
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "filename": filename,
                    "faceName": face_name,
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
                    "faceName": face_name,
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
                    "faceName": face_name,
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


# ── Face CRUD ──────────────────────────────────────────────────────


@router.get("/faces", response_model=list[SuspectProfileOut])
async def list_faces(
    tenant_id: int = Query(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    List all registered face profiles for the given tenant.
    Embeddings are NEVER returned — metadata only.
    """
    result = await session.execute(
        select(
            FaceProfileModel.id,
            FaceProfileModel.face_name,
            FaceProfileModel.alias,
            FaceProfileModel.demographics,
            FaceProfileModel.created_at,
            FaceProfileModel.tenant_id,
        ).where(FaceProfileModel.tenant_id == tenant_id).order_by(FaceProfileModel.id.desc())
    )
    rows = result.all()
    return [
        SuspectProfileOut(
            id=r.id,
            face_name=r.face_name,
            alias=r.alias,
            demographics=r.demographics,
            created_at=r.created_at.isoformat() if r.created_at else "",
            tenant_id=r.tenant_id,
        )
        for r in rows
    ]


@router.get("/faces/{face_id}", response_model=SuspectProfileOut)
async def get_face(
    face_id: int,
    tenant_id: int = Query(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """Get a single face profile by ID (metadata only)."""
    result = await session.execute(
        select(
            FaceProfileModel.id,
            FaceProfileModel.face_name,
            FaceProfileModel.alias,
            FaceProfileModel.demographics,
            FaceProfileModel.created_at,
            FaceProfileModel.tenant_id,
        ).where(FaceProfileModel.id == face_id, FaceProfileModel.tenant_id == tenant_id)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Face not found")
    return SuspectProfileOut(
        id=row.id,
        face_name=row.face_name,
        alias=row.alias,
        demographics=row.demographics,
        created_at=row.created_at.isoformat() if row.created_at else "",
        tenant_id=row.tenant_id,
    )


@router.patch("/faces/{face_id}", response_model=SuspectProfileOut)
async def update_face(
    face_id: int,
    body: SuspectUpdateIn,
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Update a face's metadata (name / alias / demographics).
    The face embedding is never modified via this endpoint.
    """
    result = await session.execute(
        select(FaceProfileModel).where(FaceProfileModel.id == face_id, FaceProfileModel.tenant_id == tenant_id)
    )
    face = result.scalar_one_or_none()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    if body.face_name is not None:
        face.face_name = body.face_name.strip()
    if body.alias is not None:
        face.alias = body.alias.strip() or None
    if body.demographics is not None:
        face.demographics = body.demographics

    await session.commit()
    await session.refresh(face)

    return SuspectProfileOut(
        id=face.id,
        face_name=face.face_name,
        alias=face.alias,
        demographics=face.demographics,
        created_at=face.created_at.isoformat() if face.created_at else "",
        tenant_id=face.tenant_id,
    )


@router.delete("/faces/{face_id}", status_code=204)
async def delete_face(
    face_id: int,
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Hard-delete a face profile.
    Logs a REGISTER_DELETE audit entry BEFORE deleting (audit log is append-only).
    Alerts referencing this face are left untouched (operator must dismiss manually).
    """
    result = await session.execute(
        select(
            FaceProfileModel.id,
            FaceProfileModel.face_name,
        ).where(FaceProfileModel.id == face_id, FaceProfileModel.tenant_id == tenant_id)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Face not found")

    # Write audit entry BEFORE delete (audit log is append-only)
    await add_audit_entry(
        session,
        event_type="REGISTER_DELETE",
        query_hash=compute_query_hash(str(face_id).encode()),
        result_name=None,   # no PII in logs — only face_id via query_hash
        distance=None,
        gps_lat=None,
        gps_lon=None,
    )

    await session.execute(
        sa_delete(FaceProfileModel).where(FaceProfileModel.id == face_id, FaceProfileModel.tenant_id == tenant_id)
    )
    await session.commit()
    # 204 No Content


# ── Face Re-enrolment ───────────────────────────────────────────


@router.put("/faces/{face_id}/face", response_model=ReenrollResponse)
async def reenroll_face(
    face_id: int,
    file: UploadFile = File(..., description="New face image (JPEG/PNG, ≤ 5 MB)"),
    tenant_id: int = Form(1, ge=1, description="Tenant identifier (default: 1)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Re-enrol a face — replace the stored 512-d ArcFace embedding
    with one freshly extracted from the uploaded image.

    Only the ``face_embedding`` and ``face_embedding_enc`` columns are
    updated; all metadata (name, alias, demographics) is left untouched.
    No raw image is persisted (AGENTS.md §9 — no raw image storage).

    A ``REGISTER_REENROLL`` event is written to the append-only audit log
    so operators have a full trail of every embedding replacement.

    Possible response statuses
    --------------------------
    ``RE_ENROLLED``  — embedding updated successfully.
    ``ERROR``        — face detection, extraction, or DB update failed;
                       the existing embedding is left unchanged.
    """
    # ── Validate image ────────────────────────────────────────────
    content, err = await _validate_file(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    # ── Confirm face exists ───────────────────────────────────────
    row = await session.execute(
        select(
            FaceProfileModel.id,
            FaceProfileModel.face_name,
        ).where(FaceProfileModel.id == face_id, FaceProfileModel.tenant_id == tenant_id)
    )
    face_row = row.one_or_none()
    if not face_row:
        raise HTTPException(status_code=404, detail="Face not found")

    face_name: str = face_row.face_name

    # ── Re-enrol (stages 1-4 + DB update) ────────────────────────
    file.file.seek(0)
    result = await reenroll_pipeline(file, session, face_id, face_name, tenant_id=tenant_id)

    if result["status"] == "ERROR":
        raise HTTPException(status_code=422, detail=result.get("error", "Re-enrolment failed"))

    from datetime import datetime, timezone
    updated_at = datetime.now(timezone.utc).isoformat()

    return ReenrollResponse(
        status=result["status"],
        profile_id=result["profile_id"],
        query_hash=result["query_hash"],
        embedding_dim=result.get("embedding_dim"),
        updated_at=updated_at,
    )


# ── Face Test Image Retrieval ─────────────────────────────────────

from fastapi.responses import FileResponse
import os

@router.get("/faces/image/{name}")
async def get_face_test_image(
    name: str,
    _user: dict = Depends(get_current_user),
):
    """
    Retrieve face image from testdata directory for demonstration.
    Does not violate database raw image storage rules.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    testdata_dir = os.path.join(base_dir, "testdata")
    
    if not os.path.exists(testdata_dir):
        raise HTTPException(status_code=404, detail="Testdata directory not found")
        
    # Standardise name: replace spaces with underscores, lower case
    clean_name = name.strip().replace(" ", "_").lower()
    
    # List files in testdata
    for filename in os.listdir(testdata_dir):
        stem, ext = os.path.splitext(filename)
        if stem.replace(" ", "_").lower() == clean_name:
            file_path = os.path.join(testdata_dir, filename)
            return FileResponse(file_path)
            
    # Also try replacing dashes with underscores
    clean_name_underscores = clean_name.replace("-", "_")
    for filename in os.listdir(testdata_dir):
        stem, ext = os.path.splitext(filename)
        std_stem = stem.replace(" ", "_").replace("-", "_").lower()
        if std_stem == clean_name_underscores:
            file_path = os.path.join(testdata_dir, filename)
            return FileResponse(file_path)
            
    raise HTTPException(status_code=404, detail="Face image not found in testdata")

