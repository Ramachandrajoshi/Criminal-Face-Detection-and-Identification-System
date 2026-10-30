"""
Pose-aware embedding — stages 1–4 of an ONNX engine's pipeline plus head pose.

Backs the ``/api/v1/pose/*`` endpoints. It is deliberately separate from
``app/core/pipeline.py``: the existing register / search flows are untouched.
Only the ONNX engines have a pose path, because the head pose is computed from
their detector's 5 landmarks: version 4 (InsightFace) and version 5
(YOLOv5-Face + AdaFace). New pose enrollments use ``POSE_EMBEDDING_VERSION``.
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.config import ADAFACE_EMBEDDING_VERSION, INSIGHTFACE_EMBEDDING_VERSION, POSE_EMBEDDING_VERSION, settings
from app.core.debug_dump import dump_detection, dump_note, dump_stage
from app.core.pose import HeadPose, estimate_pose

logger = logging.getLogger(__name__)

# version → (engine module, engine name, detector, debug dump stage prefix)
_ENGINES = {
    INSIGHTFACE_EMBEDDING_VERSION: ("app.core.face_engine", "insightface", "scrfd", "pose"),
    ADAFACE_EMBEDDING_VERSION: ("app.core.adaface_engine", "adaface", "yolov5face", "pose_v5"),
}


def _get_engine(version: int):
    return importlib.import_module(_ENGINES[version][0]).get_engine()


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


def estimate_frame_pose(image_bytes: bytes, version: int = POSE_EMBEDDING_VERSION) -> Optional[PoseEstimate]:
    """
    Detection + pose only — no embedding. Cheap enough for the webcam
    enrollment screen to call several times a second for live guidance, so it
    picks the face the way enrollment does. Returns None when no face is found.
    """
    import cv2

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    best, _ = _get_engine(version).detect_primary(frame, prefer_center=True)
    if best is None:
        return None

    pose = estimate_pose(best.landmarks)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in best.bbox)
    bbox = (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))
    return PoseEstimate(pose, best.score, bbox, pose_rejection_reason(pose))


def embed_with_pose(
    image_bytes: bytes,
    enrollment: bool = False,
    version: int = POSE_EMBEDDING_VERSION,
    enforce_limits: bool = True,
) -> Optional[PoseEmbedding]:
    """
    Detect the primary face, estimate its pose, and embed it if the pose is
    within limits. ``enrollment`` picks the centred face, otherwise the
    largest fully visible one (see ``select_primary_face``). ``version`` is
    the ONNX engine to use (4 or 5).

    Returns None when no face is found. A face outside the pose limits comes
    back with ``rejected_reason`` set, so callers can report the measured
    pose, and with ``embedding=None`` unless ``enforce_limits`` is False (a
    search embeds under every version and lets one of them decide). Engine
    failures raise.
    """
    import cv2

    _, name, detector, prefix = _ENGINES[version]
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("Failed to decode image bytes")
        return None

    engine = _get_engine(version)
    best, route = engine.detect_primary(frame, prefer_center=enrollment)
    if best is None:
        dump_note(f"{name}_detect", "no face above det threshold")
        return None

    pose = estimate_pose(best.landmarks)
    pose_meta = {
        "yaw": round(pose.yaw, 1), "pitch": round(pose.pitch, 1),
        "roll": round(pose.roll, 1), "bucket": pose.bucket,
    }
    dump_detection(
        f"{prefix}_01_detect", frame, best.bbox, best.landmarks,
        label=f"{detector} {best.score:.2f} {pose.bucket}", score=best.score, detect_pass=route, **pose_meta,
    )
    logger.debug(
        "Pose (v%d): yaw=%.1f pitch=%.1f roll=%.1f bucket=%s score=%.2f",
        version, pose.yaw, pose.pitch, pose.roll, pose.bucket, best.score,
    )

    reason = pose_rejection_reason(pose)
    if reason:
        dump_note("pose_rejected", reason)
        if enforce_limits:
            return PoseEmbedding(None, pose, best.score, reason)

    # Alignment removes roll; yaw and pitch remain and are covered by
    # enrolling several poses per person.
    aligned = engine.align(frame, best.landmarks)
    dump_stage(f"{prefix}_02_aligned_112", aligned, **pose_meta)
    embedding = engine.embed(aligned)
    return PoseEmbedding(embedding, pose, best.score, reason)
