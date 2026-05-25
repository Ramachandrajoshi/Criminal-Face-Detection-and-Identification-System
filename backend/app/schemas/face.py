"""
Pydantic request/response schemas.
All response models output camelCase JSON to match the TypeScript client types.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- Request Schemas ----------

class RegisterRequest(BaseModel):
    suspect_name: str = Field(..., min_length=1, max_length=100, description="Suspect full name")
    alias: Optional[str] = Field(None, max_length=100, description="Known alias")
    demographics: Optional[dict] = Field(None, description="Age band, gender, ethnicity")


class SearchRequest(BaseModel):
    gps_lat: Optional[float] = Field(None, ge=-90, le=90, description="GPS latitude of capture")
    gps_lon: Optional[float] = Field(None, ge=-180, le=180, description="GPS longitude of capture")


class ConfirmRequest(BaseModel):
    confirmed: bool = Field(..., description="True = CONFIRMED, False = DISMISSED")


# ---------- Response Schemas (camelCase JSON keys) ----------

def _to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split('_')
    if len(parts) == 1:
        return parts[0]
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


class _BaseCamel(BaseModel):
    """Mixin that forces camelCase JSON output."""
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
    )


class MatchResult(_BaseCamel):
    id: int
    suspect_name: str
    alias: Optional[str] = None
    distance: float


class SearchResponse(_BaseCamel):
    status: str  # "MATCH" | "NO_MATCH" | "SPOOF_BLOCKED" | "ERROR"
    query_hash: str
    matches: list[MatchResult] = []
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    match_threshold: Optional[float] = None
    alert_id: Optional[int] = None


class RegisterResponse(_BaseCamel):
    status: str  # "REGISTERED" | "ERROR"
    profile_id: Optional[int] = None
    query_hash: str
    embedding_dim: Optional[int] = None
    error: Optional[str] = None


class AlertResponse(_BaseCamel):
    id: int
    audit_log_id: Optional[int] = None
    suspect_id: Optional[int] = None
    event_type: str
    distance: Optional[float] = None
    status: str  # "PENDING_REVIEW" | "CONFIRMED" | "DISMISSED"
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    created_at: str
    confirmed_at: Optional[str] = None


class AuditEntryResponse(_BaseCamel):
    id: int
    event_type: str
    query_hash: str
    result_name: Optional[str] = None
    distance: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str = "ok"
    database: str = "connected"


class SuspectProfileOut(_BaseCamel):
    id: int
    suspect_name: str
    alias: Optional[str] = None
    demographics: Optional[dict] = None
    created_at: str   # ISO-8601 string


class SuspectUpdateIn(BaseModel):
    suspect_name: Optional[str] = Field(None, max_length=100)
    alias: Optional[str] = Field(None, max_length=100)
    demographics: Optional[dict] = None


class BatchSearchProgressEvent(_BaseCamel):
    type: str               # "start" | "progress" | "done"
    processed: int = 0
    total: int = 0
    filename: Optional[str] = None
    status: Optional[str] = None      # MATCH | NO_MATCH | SPOOF_BLOCKED | ERROR
    query_hash: Optional[str] = None
    matches: list[MatchResult] = []
    alert_id: Optional[int] = None
    elapsed_ms: int = 0
    file_ms: Optional[int] = None
    error: Optional[str] = None
