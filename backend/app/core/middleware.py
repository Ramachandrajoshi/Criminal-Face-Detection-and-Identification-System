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
    Skips /api/v1/health (public liveness probe) and Swagger/ReDoc UI paths.
    Returns a proper 401 JSONResponse so TestClient can capture it.
    """

    async def dispatch(self, request: Request, call_next):
        # Exact public endpoints — no auth required
        public_paths = (
            "/api/v1/health",
            "/api/v1/login",
            "/api/v1/token/refresh",
            "/api/v1/admin/init",
            # Swagger UI & OpenAPI schema — allow unauthenticated so devs can
            # browse the docs and use the "Authorize" button to log in.
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/api/docs/oauth2-redirect",
        )
        # Prefix-based pass-through for Swagger's bundled static assets
        # (swagger-ui JS/CSS served by FastAPI's internal StaticFiles).
        public_prefixes = (
            "/api/docs",
            "/api/redoc",
        )
        path = request.url.path
        if path in public_paths or any(path.startswith(p) for p in public_prefixes):
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
            tenant_id = user.get("tenant", 1)
            request.state.tenant_id = int(tenant_id)
            # Propagate tenant_id into the async context so DB sessions
            # (and RLS policies) use it automatically.
            from app.db.session import set_tenant_id
            set_tenant_id(int(tenant_id))
        except HTTPException:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
