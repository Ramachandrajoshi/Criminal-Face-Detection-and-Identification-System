"""
FastAPI entry point — Criminal Face Detection & Identification System.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.middleware import AuthMiddleware
from app.db.migrations.runner import run_migrations
from app.db.session import async_session_factory

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── GPU Configuration ──────────────────────────────────────────────────
def _configure_gpu() -> None:
    """
    Configure TensorFlow GPU memory growth and log available devices.

    Called once at application startup (inside lifespan).  If the host has no
    NVIDIA GPU, or ENABLE_GPU=false, this function falls back to CPU-only mode
    gracefully without raising an exception.

    Memory strategy — memory growth (not a hard VRAM cap):
      ``tf.config.experimental.set_memory_growth`` tells TF to allocate VRAM
      incrementally as tensors are created, rather than reserving the full card
      upfront.  This is the right strategy for a shared GPU server where other
      processes (e.g. llama-server) also hold VRAM.

      ``set_logical_device_configuration(memory_limit=...)`` is the alternative
      hard-cap approach, but it is **mutually exclusive** with memory growth and
      requires parsing the total VRAM size via a separate API call that expects
      a short device name (``"GPU:0"``) rather than the full path that
      ``PhysicalDevice.name`` returns (``"/physical_device:GPU:0"``).
      We therefore do NOT use it here.
    """
    if not settings.enable_gpu:
        logger.info("GPU disabled via ENABLE_GPU=false — running in CPU-only mode")
        return

    try:
        import tensorflow as tf  # imported lazily to keep startup fast on CPU hosts

        gpus = tf.config.list_physical_devices("GPU")

        if not gpus:
            logger.warning(
                "ENABLE_GPU=true but no CUDA-capable GPU was detected by TensorFlow. "
                "Check that the NVIDIA Container Toolkit is installed on the host and "
                "that the 'deploy.resources' block in docker-compose.yml is present. "
                "Falling back to CPU."
            )
            return

        # Enable per-GPU memory growth.  Must be called before any TF op runs.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # PhysicalDevice.name → "/physical_device:GPU:0"
        # Strip prefix for readable log output → "GPU:0"
        short_names = [g.name.replace("/physical_device:", "") for g in gpus]

        logger.info(
            "GPU configured: %d device(s) available — memory growth enabled "
            "(TF allocates VRAM on demand). Devices: %s",
            len(gpus),
            short_names,
        )

    except Exception as exc:  # noqa: BLE001 — never crash the server over GPU config
        logger.warning(
            "GPU configuration failed (%s). Falling back to CPU-only mode.", exc
        )


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Criminal Face Detection System starting on %s:%d", settings.api_host, settings.api_port)
    _configure_gpu()   # configure GPU/CPU before any TF model is loaded

    # ── Run pending migrations ─────────────────────────────────────
    try:
        async with async_session_factory() as session:
            summary = await run_migrations(session)
            if summary["errors"]:
                logger.error(
                    "Migration errors: %s",
                    "; ".join(e["error"] for e in summary["errors"]),
                )
            else:
                logger.info(
                    "Migrations: %d applied, %d skipped",
                    summary["applied"],
                    summary["skipped"],
                )
    except Exception as exc:
        logger.error("Migration runner failed: %s", exc)

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
