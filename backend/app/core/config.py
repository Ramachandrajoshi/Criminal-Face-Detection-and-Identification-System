"""
Configuration — loaded from environment variables.
No hard-coded secrets or thresholds.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )
    # Model
    deepface_model: str = "ArcFace"
    deepface_detector: str = "retinaface"
    match_threshold: float = 0.58

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

    # Admin login (hashed password)
    admin_username: Optional[str] = None
    admin_password_hash: Optional[str] = None
    allow_admin_init: bool = False

    # CORS (allow frontend dev server)
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()
