"""
Authentication endpoint — login, get JWT token.
"""

import logging
import secrets

import bcrypt
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


def _validate_credentials(username: str, password: str) -> bool:
    """
    Validate credentials against stored bcrypt hash.

    Raises HTTPException 500 if the stored hash looks corrupted
    (e.g. dollar-signs were expanded by the shell because the value
    was not quoted in the .env file).
    """
    if not settings.admin_username or not settings.admin_password_hash:
        logger.error("Admin credentials are not configured")
        return False

    if username != settings.admin_username:
        return False

    stored_hash = settings.admin_password_hash.strip()

    # A valid bcrypt hash always starts with $2b$ (or $2a$/$2y$) and is
    # exactly 60 characters long.  If neither is true the value was almost
    # certainly corrupted by shell variable expansion of the $ characters.
    if not (stored_hash.startswith(("$2b$", "$2a$", "$2y$")) and len(stored_hash) == 60):
        logger.error(
            "ADMIN_PASSWORD_HASH appears to be corrupted (length=%d, prefix=%r). "
            "Make sure the value is enclosed in single-quotes in your .env file "
            "and that it was generated with bcrypt (run: python backend/scripts/hash_password.py).",
            len(stored_hash),
            stored_hash[:6] if stored_hash else "",
        )
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(
            status_code=500,
            detail=(
                "Server misconfiguration: ADMIN_PASSWORD_HASH is invalid. "
                "Ensure it is quoted in .env (single-quotes) and was generated "
                "with bcrypt. See backend/scripts/hash_password.py."
            ),
        )

    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError as exc:
        logger.error("bcrypt.checkpw raised ValueError (%s) — hash may be corrupted.", exc)
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(
            status_code=500,
            detail="Server misconfiguration: bcrypt could not validate the stored password hash.",
        ) from exc


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate with username/password and receive a JWT access token.

    - Admin credentials are configured via environment variables
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

    token = create_access_token(
        {"sub": request.username, "role": "admin"},
        tenant_id=settings.default_tenant_id,
    )
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
        payload = decode_access_token(old_token, verify_exp=False)
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
    Disabled unless ALLOW_ADMIN_INIT=true.
    """
    if not settings.allow_admin_init:
        raise HTTPException(status_code=403, detail="Admin init is disabled")

    if not settings.admin_username:
        raise HTTPException(status_code=400, detail="ADMIN_USERNAME is not configured")

    return {
        "message": "Admin init is enabled; configure ADMIN_PASSWORD_HASH",
        "username": settings.admin_username,
        "password_hash_hint": "bcrypt",
        "nonce": secrets.token_hex(8),
    }
