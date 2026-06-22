"""
FastAPI entry point — Criminal Face Detection & Identification System.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.middleware import AuthMiddleware

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Criminal Face Detection System starting on %s:%d", settings.api_host, settings.api_port)
    yield
    logger.info("Criminal Face Detection System shutting down")

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Criminal Face Detection System",
    description=(
        "Decision-support platform for real-time face detection and identification.\n\n"
        "**Authentication**: Use `POST /api/v1/login` with your admin credentials to "
        "receive a Bearer token, then click **Authorize** above and paste the token."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect",
    swagger_ui_parameters={"persistAuthorization": True},
    lifespan=lifespan,
)

# ── Auth Middleware (applied to all routes except /health) ──────
app.add_middleware(AuthMiddleware)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ───────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health_check():
    """Liveness probe — no auth required."""
    from app.schemas.face import HealthResponse
    return HealthResponse(status="ok", database="connected")

# ── Routers ──────────────────────────────────────────────────────
from app.api.routes import register, search, alerts, audit, auth

app.include_router(register.router)
app.include_router(search.router)
app.include_router(alerts.router)
app.include_router(audit.router)
app.include_router(auth.router)


# ── OpenAPI security scheme (JWT Bearer) ─────────────────────────
def _add_security_scheme(openapi_schema: dict) -> dict:
    """Inject BearerAuth into the OpenAPI schema so Swagger shows a lock icon."""
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Paste the token returned by POST /api/v1/login",
    }
    # Apply globally so every endpoint shows the lock icon by default
    openapi_schema["security"] = [{"BearerAuth": []}]
    return openapi_schema


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = _add_security_scheme(schema)
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]
