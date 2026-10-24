"""
Pose-aware embedding — stages 1–4 of the InsightFace pipeline plus head pose.

Backs the ``/api/v1/pose/*`` endpoints. It is deliberately separate from
``app/core/pipeline.py``: the existing register / search flows are untouched,
and this path always uses the InsightFace engine (embedding version 4)
regardless of ``EMBEDDING_VERSION``, because its 5 landmarks are what the pose
estimate is computed from.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.debug_dump import dump_detection, dump_note, dump_stage
from app.core.pose import HeadPose, estimate_pose

logger = logging.getLogger(__name__)


@dataclass
class PoseEmbedding:
    embedding: Optional[np.ndarray]  # None when the pose was rejected
    pose: HeadPose
    quality: float                   # detector score, 0–1
    rejected_reason: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None


def pose_rejection_reason(pose: HeadPose) -> Optional[str]:
    """Why a face is turned too far to embed reliably, or None if it is usable."""
    if abs(pose.yaw) > settings.pose_max_yaw_deg:
        return f"Face turned too far sideways (yaw {pose.yaw:.0f}°, limit ±{settings.pose_max_yaw_deg:.0f}°)"
    if abs(pose.pitch) > settings.pose_max_pitch_deg:
        return f"Face tilted too far up/down (pitch {pose.pitch:.0f}°, limit ±{settings.pose_max_pitch_deg:.0f}°)"
    return None


@dataclass
class PoseEstimate:
    pose: HeadPose
    quality: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 as fractions of the frame
    rejected_reason: Optional[str]


def estimate_frame_pose(image_bytes: bytes) -> Optional[PoseEstimate]:
    """
    Detection + pose only — no embedding. Cheap enough for the webcam
    enrollment screen to call several times a second for live guidance.
    Returns None when no face is found.
    """
    import cv2

    from app.core.face_engine import get_engine

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    faces = get_engine().detect(frame)
    if not faces:
        return None

    best = max(faces, key=lambda f: f.area)
    pose = estimate_pose(best.landmarks)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in best.bbox)
    bbox = (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))
    return PoseEstimate(pose, best.score, bbox, pose_rejection_reason(pose))


def embed_with_pose(image_bytes: bytes) -> Optional[PoseEmbedding]:
    """
    Detect the largest face, estimate its pose, and embed it if the pose is
    within limits.

    Returns None when no face is found. A face outside the pose limits comes
    back with ``embedding=None`` and ``rejected_reason`` set, so callers can
    report the measured pose. Engine failures raise.
    """
    import cv2

    from app.core.face_engine import get_engine

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("Failed to decode image bytes")
        return None

    engine = get_engine()
    faces = engine.detect(frame)
    if not faces:
        dump_note("insightface_detect", "no face above det threshold")
        return None

    best = max(faces, key=lambda f: f.area)
    pose = estimate_pose(best.landmarks)
    pose_meta = {
        "yaw": round(pose.yaw, 1), "pitch": round(pose.pitch, 1),
        "roll": round(pose.roll, 1), "bucket": pose.bucket,
    }
    dump_detection(
        "pose_01_detect", frame, best.bbox, best.landmarks,
        label=f"scrfd {best.score:.2f} {pose.bucket}", score=best.score, **pose_meta,
    )
    logger.debug(
        "Pose: yaw=%.1f pitch=%.1f roll=%.1f bucket=%s score=%.2f",
        pose.yaw, pose.pitch, pose.roll, pose.bucket, best.score,
    )

    reason = pose_rejection_reason(pose)
    if reason:
        dump_note("pose_rejected", reason)
        return PoseEmbedding(None, pose, best.score, reason)

    # Alignment removes roll; yaw and pitch remain and are covered by
    # enrolling several poses per person.
    aligned = engine.align(frame, best.landmarks)
    dump_stage("pose_02_aligned_112", aligned, **pose_meta)
    embedding = engine.embed(aligned)
    return PoseEmbedding(embedding, pose, best.score)
