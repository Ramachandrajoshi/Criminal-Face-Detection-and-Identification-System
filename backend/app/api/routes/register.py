"""
Register endpoint — upload image + metadata → extract & store embedding.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.pipeline import register_pipeline
from app.core.config import settings
from app.db.session import get_session
from app.schemas.face import RegisterResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["register"])


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register_suspect(
    file: UploadFile = File(..., description="Face image (JPEG/PNG, ≤ 5 MB)"),
    suspect_name: str = Form(..., min_length=1, max_length=100, description="Suspect full name"),
    alias: Optional[str] = Form(None, max_length=100, description="Known alias"),
    demographics: Optional[str] = Form(None, description="JSON demographics (age_band, gender, ethnicity)"),
    session: AsyncSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
):
    """
    Register a new suspect profile.
    
    - Upload a clear frontal face image
    - Provide suspect name and optional metadata
    - System extracts a 512-d ArcFace embedding and stores it
    """
    # Validate file
    if not file.content_type or file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be ≤ 5 MB")

    # Rewind file so register_pipeline can read it
    file.file.seek(0)

    # Parse demographics JSON if provided
    import json
    demographics_dict = None
    if demographics:
        try:
            demographics_dict = json.loads(demographics)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid demographics JSON")

    result = await register_pipeline(
        file, session, suspect_name, alias, demographics_dict
    )

    if result["status"] == "ERROR":
        raise HTTPException(status_code=422, detail=result["error"])

    return RegisterResponse(
        status=result["status"],
        profile_id=result.get("profile_id"),
        query_hash=result["query_hash"],
        embedding_dim=result.get("embedding_dim"),
    )
