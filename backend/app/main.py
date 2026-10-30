"""
FastAPI entry point — Criminal Face Detection & Identification System.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import (
    ADAFACE_EMBEDDING_VERSION,
    CURRENT_EMBEDDING_VERSION,
    INSIGHTFACE_EMBEDDING_VERSION,
    settings,
)
from app.core.inference_executor import get_executor
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

    Memory strategy
      * TF_GPU_MEMORY_LIMIT_MB > 0 — hard cap. TF gets a logical device with
        exactly that much VRAM (reserved at startup) and can never grow past
        it. Use this to keep the project inside a fixed VRAM budget on a
        shared card; the InsightFace engine has its own separate cap
        (INSIGHTFACE_GPU_MEM_LIMIT_MB).
      * TF_GPU_MEMORY_LIMIT_MB = 0 (default) — memory growth: TF allocates
        VRAM on demand. No upper bound — on a large card TF can keep growing.

      The two are mutually exclusive in TF, so exactly one is applied.
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

        # PhysicalDevice.name → "/physical_device:GPU:0"
        # Strip prefix for readable log output → "GPU:0"
        short_names = [g.name.replace("/physical_device:", "") for g in gpus]

        # Must be configured before any TF op runs.
        if not settings.tf_enable_gpu:
            # Hides the GPU from TensorFlow only — ONNX Runtime (InsightFace)
            # creates its own CUDA context and still uses it.
            tf.config.set_visible_devices([], "GPU")
            logger.info(
                "TensorFlow GPU disabled via TF_ENABLE_GPU=false — DeepFace runs on CPU; "
                "InsightFace still uses %s",
                short_names,
            )
            return

        limit_mb = settings.tf_gpu_memory_limit_mb
        if limit_mb > 0:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                )
            logger.info(
                "GPU configured: %d device(s), TF VRAM capped at %d MB each. Devices: %s",
                len(gpus), limit_mb, short_names,
            )
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
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


# ── Model warm-up ────────────────────────────────────────────────
def _warmup_models() -> None:
    """
    Force DeepFace to load its model weights before the app accepts traffic.

    Without this, the first real request after boot pays full model-load
    latency for RetinaFace/MTCNN/OpenCV (detector) and ArcFace (embedding).
    Running this synchronously on the inference executor — and awaiting it
    to completion before ``yield`` in ``lifespan`` — also closes a benign
    cold-start race: DeepFace's internal model cache is a plain dict with no
    lock, so two concurrent first-requests could otherwise both miss the
    cache and double-load weights into GPU memory.

    Never raises: a warm-up failure should log a warning, not block startup
    (mirrors ``_configure_gpu``'s fail-open behaviour). Falls back to a
    synthetic image for detector warm-up since no real face fixture ships
    with the repo — DeepFace still builds/loads each detector backend's
    model before it can determine "no face found", so this reaches the
    weights-loaded end state without needing a real photo.
    """
    start = time.perf_counter()

    try:
        from deepface import DeepFace

        DeepFace.build_model(settings.deepface_model)
        logger.info("Warmed embedding model '%s'", settings.deepface_model)
    except Exception as exc:  # noqa: BLE001 — warm-up must never crash startup
        logger.warning("Embedding model warm-up failed (%s) — will load lazily on first request", exc)

    try:
        import cv2
        import numpy as np

        from app.core.pipeline import detect_face

        dummy = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", dummy)
        if ok:
            detect_face(buf.tobytes())  # discards result; only weight-loading matters here
            logger.info("Warmed detector backends")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Detector warm-up failed (%s) — will load lazily on first request", exc)

    onnx_engines = {
        INSIGHTFACE_EMBEDDING_VERSION: ("InsightFace", "app.core.face_engine"),
        ADAFACE_EMBEDDING_VERSION: ("AdaFace", "app.core.adaface_engine"),
    }
    if CURRENT_EMBEDDING_VERSION in onnx_engines:
        name, module = onnx_engines[CURRENT_EMBEDDING_VERSION]
        try:
            import importlib

            import numpy as np

            get_engine = importlib.import_module(module).get_engine

            # Loads (and on first run downloads) the ONNX models pre-traffic.
            engine_start = time.perf_counter()
            engine = get_engine()
            # One detect + embed pass on blank input: the first GPU inference
            # pays CUDA/cuDNN initialisation and, on GPUs newer than the ONNX
            # Runtime build (e.g. Blackwell), about a minute of kernel JIT
            # compilation — cached by the driver (CUDA_CACHE_PATH) afterwards.
            engine.detect(np.zeros((engine.det_size, engine.det_size, 3), dtype=np.uint8))
            # The "large" detection fallback runs at twice the input size.
            engine.detect(np.zeros((engine.det_size, engine.det_size, 3), dtype=np.uint8), size=2 * engine.det_size)
            engine.embed(np.zeros((112, 112, 3), dtype=np.uint8))
            logger.info(
                "Warmed %s engine in %.0fms (providers=%s)",
                name, (time.perf_counter() - engine_start) * 1000, engine.providers,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s warm-up failed (%s) — will load lazily on first request", name, exc)

    logger.info("Model warm-up finished in %.0fms", (time.perf_counter() - start) * 1000)


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Criminal Face Detection System starting on %s:%d", settings.api_host, settings.api_port)
    _configure_gpu()   # configure GPU/CPU before any TF model is loaded

    # Warm up DeepFace models on the same executor used at request time, so
    # the first-ever model load happens pre-traffic on an "inference" thread
    # rather than lazily on whichever thread handles the first request.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(get_executor(), _warmup_models)

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
from app.api.routes import register, search, alerts, audit, auth, liveness, pose

app.include_router(register.router)
app.include_router(search.router)
app.include_router(alerts.router)
app.include_router(audit.router)
app.include_router(auth.router)
app.include_router(liveness.router)
app.include_router(pose.router)


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
