"""
Unit tests for head-pose estimation (app/core/pose.py), the pose quality gate
(app/core/pose_pipeline.py) and the pose ANN SQL (app/db/pose_ops.py).
"""

import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.core.pose import HeadPose, estimate_pose, pose_bucket

# A synthetic 3D face, x right, y down, z toward the camera. Inter-eye and
# eye-to-mouth distances are equal and the nose tip sits half of that in front
# of the face plane — the model estimate_pose assumes — so its estimates
# should come back exact up to rounding.
_D = 65.0
_FACE_3D = np.array([
    [-_D / 2, 0.0, 0.0],               # left eye
    [_D / 2, 0.0, 0.0],                # right eye
    [0.0, 0.4949 * _D, _D / 2],        # nose tip
    [-0.35 * _D, _D, 0.0],             # left mouth corner
    [0.35 * _D, _D, 0.0],              # right mouth corner
])


def _project(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Orthographic image landmarks of the synthetic face at a given pose."""
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    rot_yaw = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rot_pitch = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    pts = (_FACE_3D @ rot_yaw.T @ rot_pitch.T)[:, :2]
    rot_roll = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return pts @ rot_roll.T + np.array([320.0, 240.0])


def test_arcface_template_is_frontal():
    from app.core.face_engine import ARCFACE_TEMPLATE

    pose = estimate_pose(ARCFACE_TEMPLATE)
    assert abs(pose.yaw) < 1 and abs(pose.pitch) < 1 and abs(pose.roll) < 1
    assert pose.bucket == "frontal"


@pytest.mark.parametrize("yaw", [-60, -30, 0, 30, 60])
def test_yaw_is_recovered(yaw):
    pose = estimate_pose(_project(yaw=yaw))
    assert pose.yaw == pytest.approx(yaw, abs=1)
    assert pose.pitch == pytest.approx(0, abs=1)


@pytest.mark.parametrize("pitch", [-30, -15, 15, 30])
def test_pitch_is_recovered(pitch):
    pose = estimate_pose(_project(pitch=pitch))
    assert pose.pitch == pytest.approx(pitch, abs=2)
    assert pose.yaw == pytest.approx(0, abs=1)


@pytest.mark.parametrize("roll", [-25, 20])
def test_in_plane_rotation_does_not_change_yaw_or_pitch(roll):
    # Roll is measured as the eye-line angle in the image. A head that is both
    # turned and tilted already has a sloped eye line, so compare against the
    # same head without in-plane rotation rather than against `roll` itself.
    upright = estimate_pose(_project(yaw=30, pitch=15))
    rotated = estimate_pose(_project(yaw=30, pitch=15, roll=roll))
    assert rotated.roll - upright.roll == pytest.approx(roll, abs=0.5)
    assert rotated.yaw == pytest.approx(upright.yaw, abs=0.5)
    assert rotated.pitch == pytest.approx(upright.pitch, abs=0.5)


@pytest.mark.parametrize("yaw,pitch", [(30, 15), (-40, -20), (20, 25)])
def test_combined_yaw_and_pitch_stay_in_the_right_bucket(yaw, pitch):
    # Combined rotations couple the axes a little; the estimate only has to be
    # good enough to bucket the face and apply the pose limits.
    pose = estimate_pose(_project(yaw=yaw, pitch=pitch))
    assert pose.yaw == pytest.approx(yaw, abs=8)
    assert pose.pitch == pytest.approx(pitch, abs=8)
    assert pose.bucket == pose_bucket(yaw, pitch)


def test_degenerate_landmarks_do_not_raise():
    assert estimate_pose(np.zeros((5, 2))) == HeadPose(0.0, 0.0, 0.0)


@pytest.mark.parametrize("yaw,pitch,expected", [
    (0, 0, "frontal"),
    (-14, 19, "frontal"),
    (25, 0, "right_3q"),
    (-40, 0, "left_3q"),
    (50, 0, "right_profile"),
    (-70, -25, "left_profile_down"),
    (5, 22, "frontal_up"),
])
def test_pose_bucket(yaw, pitch, expected):
    assert pose_bucket(yaw, pitch) == expected


@pytest.mark.parametrize("pose,rejected", [
    (HeadPose(0, 0, 0), False),
    (HeadPose(-79, 0, 0), False),
    (HeadPose(85, 0, 0), True),
    (HeadPose(0, -55, 0), True),
])
def test_pose_rejection(pose, rejected):
    from app.core.pose_pipeline import pose_rejection_reason

    assert (pose_rejection_reason(pose) is not None) == rejected


def test_embed_with_pose_skips_embedding_for_rejected_pose(monkeypatch):
    import cv2

    from app.core import face_engine, pose_pipeline
    from app.core.face_engine import DetectedFace

    engine = MagicMock()
    engine.detect_primary.return_value = (
        DetectedFace(np.array([0, 0, 100, 100.0]), 0.9, _project(yaw=88)), "standard",
    )
    monkeypatch.setattr(face_engine, "get_engine", lambda: engine)
    _, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))

    result = pose_pipeline.embed_with_pose(buf.tobytes())

    assert result.embedding is None and not result.accepted
    assert "yaw" in result.rejected_reason
    engine.embed.assert_not_called()


@pytest.mark.asyncio
async def test_pose_ann_query_is_one_statement_grouped_per_face():
    from app.db.pose_ops import pose_ann_query

    session = AsyncMock()
    session.execute.return_value = [(5, "Jane", None, 0.21, "left_3q", -31.0, 2.0)]

    matches = await pose_ann_query(
        session, np.full(512, 0.5, np.float32), threshold=0.6, limit=3, tenant_id=7, embedding_version=4
    )

    session.execute.assert_awaited_once()
    sql, params = session.execute.call_args.args
    statement = str(sql)
    assert "DISTINCT ON (p.face_id)" in statement
    assert "p.tenant_id = :tenant_id" in statement
    assert "<= :threshold" in statement
    assert params["tenant_id"] == 7 and params["limit"] == 3 and params["threshold"] == 0.6
    assert params["embedding_version"] == 4
    assert matches == [{
        "id": 5, "face_name": "Jane", "alias": None, "distance": 0.21,
        "matched_pose": "left_3q", "matched_yaw": -31.0, "matched_pitch": 2.0, "tenant_id": 7,
    }]


def _engine_with(face, embedding=None):
    engine = MagicMock()
    engine.detect_primary.return_value = (face, "none" if face is None else "standard")
    engine.align.return_value = np.zeros((112, 112, 3), np.uint8)
    engine.embed.return_value = embedding
    return engine


def _png(h=480, w=640):
    import cv2

    return cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))[1].tobytes()


def test_frame_pose_normalises_and_clips_the_box(monkeypatch):
    from app.core import face_engine, pose_pipeline
    from app.core.face_engine import DetectedFace

    engine = _engine_with(DetectedFace(np.array([-32.0, 48, 320, 528]), 0.88, _project(yaw=5)))
    monkeypatch.setattr(face_engine, "get_engine", lambda: engine)

    estimate = pose_pipeline.estimate_frame_pose(_png())

    assert estimate.bbox == (0.0, 0.1, 0.5, 1.0)       # x1 < 0 and y2 > height are clipped
    assert estimate.quality == 0.88 and estimate.rejected_reason is None
    assert abs(estimate.pose.yaw - 5) < 1
    assert engine.detect_primary.call_args.kwargs["prefer_center"] is True   # picks like enrollment


@pytest.mark.parametrize("image, face", [(b"not an image", None), ("blank", None)])
def test_frame_pose_without_a_face_is_none(monkeypatch, image, face):
    from app.core import face_engine, pose_pipeline

    monkeypatch.setattr(face_engine, "get_engine", lambda: _engine_with(face))
    assert pose_pipeline.estimate_frame_pose(_png() if image == "blank" else image) is None


@pytest.mark.parametrize("image", [b"not an image", "blank"])
def test_embed_with_pose_without_a_face_is_none(monkeypatch, image):
    from app.core import face_engine, pose_pipeline

    engine = _engine_with(None)
    monkeypatch.setattr(face_engine, "get_engine", lambda: engine)
    assert pose_pipeline.embed_with_pose(_png() if image == "blank" else image) is None
    engine.embed.assert_not_called()


def test_embed_with_pose_embeds_an_accepted_pose(monkeypatch):
    from app.core import face_engine, pose_pipeline
    from app.core.face_engine import DetectedFace

    embedding = np.full(512, 1 / np.sqrt(512), np.float32)
    engine = _engine_with(DetectedFace(np.array([100.0, 100, 300, 300]), 0.93, _project(yaw=-30)), embedding)
    monkeypatch.setattr(face_engine, "get_engine", lambda: engine)

    result = pose_pipeline.embed_with_pose(_png(), enrollment=True)

    assert result.accepted and result.embedding is embedding
    assert result.pose.bucket == pose_bucket(result.pose.yaw, result.pose.pitch)
    assert engine.detect_primary.call_args.kwargs["prefer_center"] is True


def test_pose_version_follows_embedding_version_for_onnx_engines():
    """New poses use version 5 when EMBEDDING_VERSION=5, otherwise version 4."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    for env_version, pose_version in (("3", "4"), ("4", "4"), ("5", "5")):
        proc = subprocess.run(
            [sys.executable, "-c", "import app.core.config as c; print(c.POSE_EMBEDDING_VERSION)"],
            env={**os.environ, "EMBEDDING_VERSION": env_version}, capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == pose_version


def test_embed_with_pose_version_5_uses_the_adaface_engine(monkeypatch):
    from app.core import adaface_engine, face_engine, pose_pipeline
    from app.core.face_engine import DetectedFace

    embedding = np.full(512, 1 / np.sqrt(512), np.float32)
    engine = _engine_with(DetectedFace(np.array([100.0, 100, 300, 300]), 0.91, _project(yaw=40)), embedding)
    monkeypatch.setattr(adaface_engine, "get_engine", lambda: engine)
    monkeypatch.setattr(face_engine, "get_engine", lambda: pytest.fail("version 4 engine used"))
    detections, stages = [], []
    monkeypatch.setattr(pose_pipeline, "dump_detection", lambda tag, *a, **k: detections.append((tag, k["label"])))
    monkeypatch.setattr(pose_pipeline, "dump_stage", lambda tag, *a, **k: stages.append(tag))

    result = pose_pipeline.embed_with_pose(_png(), False, 5)

    assert result.accepted and result.embedding is embedding
    assert result.pose.bucket == "right_3q"
    assert detections[0][0] == "pose_v5_01_detect" and detections[0][1].startswith("yolov5face")
    assert stages == ["pose_v5_02_aligned_112"]


def test_frame_pose_version_5_uses_the_adaface_engine(monkeypatch):
    from app.core import adaface_engine, pose_pipeline
    from app.core.face_engine import DetectedFace

    engine = _engine_with(DetectedFace(np.array([64.0, 48, 320, 432]), 0.8, _project(yaw=-20)))
    monkeypatch.setattr(adaface_engine, "get_engine", lambda: engine)

    estimate = pose_pipeline.estimate_frame_pose(_png(), 5)

    assert estimate.bbox == (0.1, 0.1, 0.5, 0.9)
    assert estimate.pose.bucket == "left_3q"


def test_embed_with_pose_can_embed_beyond_the_limits(monkeypatch):
    """A search embeds under every version and lets one decide on the pose."""
    from app.core import face_engine, pose_pipeline
    from app.core.face_engine import DetectedFace

    embedding = np.full(512, 1 / np.sqrt(512), np.float32)
    engine = _engine_with(DetectedFace(np.array([0, 0, 100, 100.0]), 0.9, _project(yaw=88)), embedding)
    monkeypatch.setattr(face_engine, "get_engine", lambda: engine)

    result = pose_pipeline.embed_with_pose(_png(), False, 4, enforce_limits=False)

    assert not result.accepted and "yaw" in result.rejected_reason
    assert result.embedding is embedding
