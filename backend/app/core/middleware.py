"""
Authentication middleware — attaches user info to request state.
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.auth import decode_access_token


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates JWT on every request.
    Skips /api/v1/health (public liveness probe).
    Returns a proper 401 JSONResponse so TestClient can capture it.
    """

    async def dispatch(self, request: Request, call_next):
        # Public endpoints — no auth required
        public_paths = (
            "/api/v1/health",
            "/api/v1/login",
            "/api/v1/token/refresh",
            "/api/v1/admin/init",
        )
        if request.url.path in public_paths:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1]
        try:
            user = decode_access_token(token)
            request.state.user = user
        except HTTPException:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
