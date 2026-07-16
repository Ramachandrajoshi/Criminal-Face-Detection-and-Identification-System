"""
Configuration — loaded from environment variables.
No hard-coded secrets or thresholds.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # CORS (allow frontend dev server)
    allowed_origins: list[str] = ["*"]


settings = Settings()
