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
    description="Decision-support platform for real-time face detection and identification",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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
