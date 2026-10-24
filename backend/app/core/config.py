"""
Configuration — loaded from environment variables.
No hard-coded secrets or thresholds.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Every embedding recipe the pipeline can still reproduce, newest first.
# Stored vectors are only ever compared against a query embedded with their
# own recipe, so a recipe stays listed here for as long as any row still
# carries that ``embedding_version``.
#
#   1–3 — DeepFace ArcFace + the ``_preprocess_for_arcface`` recipes
#   4   — InsightFace buffalo_l on ONNX Runtime (app/core/face_engine.py)
#
# Adding a DeepFace recipe means: add a branch in ``_preprocess_for_arcface``
# and list it in both tuples below.
SUPPORTED_EMBEDDING_VERSIONS = (4, 3, 2, 1)
DEEPFACE_EMBEDDING_VERSIONS = (3, 2, 1)
INSIGHTFACE_EMBEDDING_VERSION = 4

assert set(SUPPORTED_EMBEDDING_VERSIONS) == {*DEEPFACE_EMBEDDING_VERSIONS, INSIGHTFACE_EMBEDDING_VERSION}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.docker"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    # Model
    deepface_model: str = "ArcFace"
    deepface_detector: str = "retinaface"
    match_threshold: float = 0.58

    # Recipe used for new enrollments and re-enrollments (EMBEDDING_VERSION).
    # Existing vectors keep the recipe they were enrolled with until they are
    # re-enrolled, and search keeps matching them under that recipe — so
    # switching this never invalidates already-registered faces.
    #   3 — DeepFace ArcFace (default)
    #   4 — InsightFace buffalo_l (research licence; see face_engine.py)
    embedding_version: int = 3

    # --- InsightFace (embedding version 4) ---
    # Cosine-distance match threshold for version-4 vectors. A different model
    # has a different distance distribution, so MATCH_THRESHOLD (versions 1–3)
    # does not carry over. PROVISIONAL: calibrate on your own data with
    # backend/scripts/benchmark_engines.py and the bias-audit notebook
    # (AGENTS.md §8) before relying on it.
    match_threshold_insightface: float = 0.60
    insightface_model_dir: str = "~/.insightface/models/buffalo_l"
    # Download the pinned buffalo_l pack on first use if files are missing.
    insightface_auto_download: bool = True
    # SCRFD input size (square) and minimum face score.
    insightface_det_size: int = 640
    insightface_det_threshold: float = 0.5
    # Upper bound on the ONNX Runtime CUDA memory arena.
    insightface_gpu_mem_limit_mb: int = 1536

    # --- Pose-aware enrollment & search (/api/v1/pose/*) ---
    # Always InsightFace (version 4), independent of EMBEDDING_VERSION.
    # Faces turned further than this are rejected: past ~80° yaw the eye and
    # mouth landmarks are guesses and the embedding is not trustworthy.
    pose_max_yaw_deg: float = 80.0
    pose_max_pitch_deg: float = 50.0
    # Upper bound on stored pose embeddings per face profile.
    pose_max_per_face: int = 20

    # ArcFace embedding quality settings
    # normalization must match the backbone's training recipe:
    #   "ArcFace"  → (pixel − 127.5) / 128   ← correct for ArcFace/InsightFace
    #   "base"     → pixel / 255              ← wrong for ArcFace (default in DeepFace)
    # Never change this without re-registering all suspect embeddings.
    arcface_normalization: str = "ArcFace"
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) normalises
    # per-image illumination on the luminance channel before embedding.
    # Disabling it may slightly increase same-person cosine distance.
    enable_clahe: bool = True
    # Minimum CLAHE clip limit (lower = less aggressive contrast enhancement).
    clahe_clip_limit: float = 2.0

    # --- Enhanced preprocessing stages (all applied BEFORE the 112×112 resize) ---

    # Upscale tiny face crops via Lanczos super-sampling before any processing.
    # CCTV sub-crops are often < 80 px wide; upsampling to a minimum side length
    # before enhancement gives CLAHE and the unsharp mask more pixels to work on.
    enable_upscale_small_crops: bool = True
    # Minimum side length (px) below which a crop is considered "small" and
    # will be Lanczos-upscaled to this size before further processing.
    small_crop_min_side: int = 160

    # Non-local means denoising on the colour image.
    # Applied before CLAHE so the contrast equaliser does not amplify sensor
    # noise. h and hColor are the filter strengths; keep low (5) to avoid
    # over-blurring fine facial features.
    enable_denoising: bool = True

    # Auto-gamma correction via mean luminance.
    # Computes gamma = log(0.5) / log(mean/255) and applies x^gamma to the LUT.
    # For dark images (mean ≈ 40): gamma ≈ 0.32 → x^0.32 lifts shadows.
    # For mid-tone images (mean ≈ 128): gamma ≈ 1.0 → identity (no-op).
    enable_gamma_correction: bool = True

    # Unsharp masking — mild high-frequency boost applied after CLAHE.
    # Recovers edge detail that denoising slightly blurs.
    # strength=1.5 means: output = 1.5*sharp − 0.5*blurred (net +50% edges).
    enable_unsharp_mask: bool = True
    unsharp_strength: float = 1.5

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "criminaldb"
    postgres_user: str = "appuser"
    postgres_password: str = Field(..., min_length=1)
    db_encryption_key: str = Field(..., min_length=1)

    # API
    jwt_secret: str = Field(..., min_length=1)
    jwt_expiry_hours: int = 8
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    default_tenant_id: int = 1

    # Admin login (hashed password)
    admin_username: Optional[str] = None
    admin_password_hash: Optional[str] = None
    allow_admin_init: bool = False

    # GPU acceleration
    # Set ENABLE_GPU=false in .env to force CPU-only mode (e.g. on machines
    # without a CUDA-capable GPU or NVIDIA Container Toolkit).
    enable_gpu: bool = True
    # Let TensorFlow (DeepFace: recipes 1–3, detector warm-up) use the GPU.
    # Set false when the image's TensorFlow cannot run on the card — TF 2.16
    # (CUDA 12.3 / cuDNN 8.9) fails on RTX 50-series / Blackwell GPUs (compute
    # capability 12.0) with CUDA_ERROR_INVALID_HANDLE. DeepFace then runs on
    # CPU while ONNX Runtime (InsightFace: version 4 and /pose) keeps the GPU.
    tf_enable_gpu: bool = True
    # Hard VRAM cap for TensorFlow (DeepFace models), in MB. 0 = no cap
    # (memory growth). The InsightFace engine is capped separately by
    # INSIGHTFACE_GPU_MEM_LIMIT_MB; both share one CUDA context (~0.3–0.5 GB).
    tf_gpu_memory_limit_mb: int = 0

    # Size of the dedicated thread pool used to offload blocking DeepFace/OpenCV
    # calls off the event loop (see app/core/inference_executor.py). Bounds how
    # many detect/embed calls run concurrently against the shared GPU. Tune via
    # load testing (watch nvidia-smi + p95 latency) rather than guessing high.
    inference_pool_size: int = 3
    # Max concurrent per-file pipeline calls inside batch SSE endpoints. Should
    # generally track inference_pool_size since each concurrent batch file
    # consumes one inference-pool slot.
    batch_pipeline_concurrency: int = 3

    # --- Debug: per-stage face dumps (app/core/debug_dump.py) ---
    # Writes the face image after every pipeline stage, plus a manifest of
    # lighting stats, to one folder per request under DEBUG_FACE_DUMP_DIR.
    # DEBUG ONLY — this persists raw face images (AGENTS.md §15). Never enable
    # it in production; delete the folder when done.
    debug_face_dump: bool = False
    # Relative paths resolve against the working directory (/app in Docker).
    debug_face_dump_dir: str = "debug_faces"

    # CORS (allow frontend dev server)
    allowed_origins: list[str] = ["*"]


settings = Settings()

if settings.embedding_version not in SUPPORTED_EMBEDDING_VERSIONS:
    raise ValueError(
        f"EMBEDDING_VERSION={settings.embedding_version} is not one of "
        f"{SUPPORTED_EMBEDDING_VERSIONS}"
    )

# Recipe used for new enrollments and re-enrollments.
CURRENT_EMBEDDING_VERSION = settings.embedding_version


def match_threshold_for_version(version: int) -> float:
    """Cosine-distance threshold for vectors of the given embedding version."""
    if version == INSIGHTFACE_EMBEDDING_VERSION:
        return settings.match_threshold_insightface
    return settings.match_threshold
