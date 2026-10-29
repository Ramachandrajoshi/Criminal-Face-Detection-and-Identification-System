"""
Schemas for the pose-aware endpoints (/api/v1/pose/*).
camelCase JSON, like app/schemas/face.py.
"""

from typing import Optional

from app.schemas.face import _BaseCamel


class PoseInfo(_BaseCamel):
    yaw: float
    pitch: float
    roll: float
    bucket: str
    quality: float


class PoseEstimateResponse(_BaseCamel):
    face_found: bool
    pose: Optional[PoseInfo] = None
    # x1, y1, x2, y2 as fractions of the frame, for drawing a box over video
    bbox: Optional[list[float]] = None
    usable: bool = False
    reason: Optional[str] = None


class PoseImageResult(_BaseCamel):
    filename: str
    status: str  # "ACCEPTED" | "REJECTED"
    pose_id: Optional[int] = None
    pose: Optional[PoseInfo] = None
    error: Optional[str] = None


class PoseRegisterResponse(_BaseCamel):
    status: str  # "REGISTERED" | "POSES_ADDED" | "POSES_REPLACED"
    profile_id: int
    accepted: int
    rejected: int
    pose_buckets: list[str]
    images: list[PoseImageResult]
    tenant_id: int = 1


class PoseOut(_BaseCamel):
    id: int
    pose_bucket: str
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    quality: Optional[float] = None
    created_at: str


class PoseMatchResult(_BaseCamel):
    id: int
    face_name: str
    alias: Optional[str] = None
    distance: float
    matched_pose: str
    matched_yaw: Optional[float] = None
    matched_pitch: Optional[float] = None
    tenant_id: int = 1


class PoseSearchResponse(_BaseCamel):
    # "MATCH" | "NO_MATCH" | "LOW_QUALITY" (pose beyond limits) | "ERROR"
    status: str
    query_hash: str
    query_pose: Optional[PoseInfo] = None
    matches: list[PoseMatchResult] = []
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    match_threshold: Optional[float] = None
    alert_id: Optional[int] = None
    error: Optional[str] = None
