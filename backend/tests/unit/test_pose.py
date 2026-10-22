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
    engine.detect.return_value = [
        DetectedFace(np.array([0, 0, 100, 100.0]), 0.9, _project(yaw=88)),
    ]
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
        session, np.full(512, 0.5, np.float32), threshold=0.6, limit=3, tenant_id=7
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
