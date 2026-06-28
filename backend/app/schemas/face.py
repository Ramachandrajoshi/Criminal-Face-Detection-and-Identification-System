"""
Pydantic request/response schemas.
All response models output camelCase JSON to match the TypeScript client types.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- Request Schemas ----------

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
    face_name: str
    alias: Optional[str] = None
    distance: float
    tenant_id: int = 1


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
    tenant_id: int = 1


class ReenrollResponse(_BaseCamel):
    """Response for PUT /faces/{face_id}/face (face re-enrolment)."""
    status: str              # "RE_ENROLLED" | "ERROR"
    profile_id: int
    query_hash: str          # SHA-256 of the new embedding bytes
    embedding_dim: Optional[int] = None
    updated_at: str          # ISO-8601 timestamp of the update
    error: Optional[str] = None


class AlertResponse(_BaseCamel):
    id: int
    audit_log_id: Optional[int] = None
    face_id: Optional[int] = None
    face_name: Optional[str] = None
    face_alias: Optional[str] = None
    event_type: str
    distance: Optional[float] = None
    status: str  # "PENDING_REVIEW" | "CONFIRMED" | "DISMISSED"
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    created_at: str
    confirmed_at: Optional[str] = None
    tenant_id: int = 1


class AuditEntryResponse(_BaseCamel):
    id: int
    event_type: str
    query_hash: str
    result_name: Optional[str] = None
    distance: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    timestamp: str
    tenant_id: int = 1


class HealthResponse(BaseModel):
    status: str = "ok"
    database: str = "connected"


class SuspectProfileOut(_BaseCamel):
    id: int
    face_name: str
    alias: Optional[str] = None
    demographics: Optional[dict] = None
    created_at: str   # ISO-8601 string
    tenant_id: int = 1


class SuspectUpdateIn(BaseModel):
    face_name: Optional[str] = Field(None, max_length=100)
    alias: Optional[str] = Field(None, max_length=100)
    demographics: Optional[dict] = None
    tenant_id: Optional[int] = None



