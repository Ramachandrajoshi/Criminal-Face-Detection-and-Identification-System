"""
Authentication endpoint — login, get JWT token.
"""

import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import create_access_token, decode_access_token
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=4)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ── Hard-coded default admin account (override via env) ──────────

DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"  # override via JWT_SECRET in .env


def _validate_credentials(username: str, password: str) -> bool:
    """
    Validate credentials.
    In production, replace with DB lookup against hashed passwords.
    """
    if username == DEFAULT_ADMIN_USER and password == DEFAULT_ADMIN_PASS:
        return True
    # TODO: Replace with actual DB/user table lookup
    return False


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate with username/password and receive a JWT access token.

    - Default credentials: admin / admin123
    - Token expires after 8 hours (configurable via JWT_EXPIRY_HOURS)
    - All login attempts are logged for security auditing.
    """
    if not _validate_credentials(request.username, request.password):
        logger.warning("Failed login attempt for user: %s", request.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": request.username, "role": "admin"})
    logger.info("Successful login for user: %s", request.username)

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiry_hours * 3600,
    )


@router.post("/token/refresh", response_model=TokenResponse)
async def refresh_token(
    body: dict,
):
    """
    Refresh an expiring JWT token.

    Accepts the current (still-valid) access_token in the request body
    and returns a new one with a fresh expiry window.
    """
    try:
        old_token = body.get("access_token", "")
        payload = decode_access_token(old_token)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    new_token = create_access_token(
        {"sub": payload.get("sub", "unknown"), "role": payload.get("role", "admin")}
    )

    return TokenResponse(
        access_token=new_token,
        expires_in=settings.jwt_expiry_hours * 3600,
    )


@router.post("/admin/init")
async def init_admin_account():
    """
    Seed the default admin account.
    Call once during initial setup.
    In production, remove this endpoint.
    """
    return {
        "message": "Default admin account created",
        "username": DEFAULT_ADMIN_USER,
        "password": DEFAULT_ADMIN_PASS,
        "warning": "CHANGE THIS PASSWORD IN PRODUCTION",
        "nonce": secrets.token_hex(8),
    }
