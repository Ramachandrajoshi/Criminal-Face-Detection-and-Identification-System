"""
Integration tests: pose-aware enrollment and search (/api/v1/pose/*).

The InsightFace engine is replaced by ``embed_with_pose`` stubs; the route
logic, validation, audit and alert wiring run for real.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.auth import create_access_token
from app.core.pose import HeadPose
from app.core.pose_pipeline import PoseEmbedding
from app.db.session import get_session
from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {create_access_token({'sub': 'operator', 'role': 'admin'})}"}


@pytest.fixture
def session():
    session = AsyncMock()

    async def override_session():
        yield session

    with patch.dict(app.dependency_overrides, {get_session: override_session}):
        yield session


def _png(name: str, shade: int = 128) -> tuple:
    buf = io.BytesIO()
    Image.new("RGB", (112, 112), color=(shade, shade, shade)).save(buf, format="PNG")
    return ("files", (name, buf.getvalue(), "image/png"))


def _embed_by_image(uploads: list[tuple], results: list) -> MagicMock:
    """
    Stub for embed_with_pose keyed on image bytes. Uploads are embedded
    concurrently on the inference pool, so call order is not file order.
    """
    by_bytes = {upload[1][1]: result for upload, result in zip(uploads, results)}
    return MagicMock(side_effect=lambda content, *args: by_bytes[content])


def _accepted(yaw: float, pitch: float = 0.0, fill: float = 0.1) -> PoseEmbedding:
    return PoseEmbedding(np.full(512, fill, np.float32), HeadPose(yaw, pitch, 0.0), 0.9)


def _rejected(yaw: float) -> PoseEmbedding:
    return PoseEmbedding(None, HeadPose(yaw, 0.0, 0.0), 0.8, "Face turned too far sideways")


# ── Registration ─────────────────────────────────────────────────


def test_register_stores_every_usable_pose_and_averages_them_as_primary(
    client, auth_headers, session
):
    left, frontal = _accepted(-35, fill=0.1), _accepted(3, fill=0.3)
    left.embedding[:256] = -0.1
    uploads = [_png("left.png", 10), _png("front.png", 20), _png("profile.png", 30), _png("blank.png", 40)]
    stub = _embed_by_image(uploads, [left, frontal, _rejected(88), None])

    with patch("app.api.routes.pose.embed_with_pose", stub), \
         patch("app.api.routes.pose.register_profile_with_poses",
               AsyncMock(return_value=(41, [100, 101]))) as store, \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=1)) as audit:
        response = client.post(
            "/api/v1/pose/register", headers=auth_headers,
            data={"face_name": "Jane Doe", "tenant_id": 7}, files=uploads,
        )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "REGISTERED"
    assert body["profileId"] == 41
    assert (body["accepted"], body["rejected"]) == (2, 2)
    assert body["poseBuckets"] == ["frontal", "left_3q"]
    assert [i["status"] for i in body["images"]] == ["ACCEPTED", "ACCEPTED", "REJECTED", "REJECTED"]
    assert [i["poseId"] for i in body["images"]] == [100, 101, None, None]
    # A rejected-by-pose image still reports what was measured.
    assert body["images"][2]["pose"]["bucket"] == "right_profile"
    assert body["images"][3]["error"] == "No face detected in the uploaded image"

    args, kwargs = store.call_args
    assert args[1] == "Jane Doe"
    # Profile embedding = normalised average of the accepted photos.
    expected = (left.embedding + frontal.embedding) / 2
    np.testing.assert_allclose(args[4], expected / np.linalg.norm(expected), rtol=1e-6)
    assert len(args[5]) == 2 and kwargs["tenant_id"] == 7
    # Uploads for enrollment pick the centred face.
    assert all(call.args[1:] == (True,) for call in stub.call_args_list)
    assert audit.call_args.kwargs["event_type"] == "REGISTER_POSES"


def test_register_fails_when_no_image_is_usable(client, auth_headers, session):
    uploads = [_png("a.png", 10), _png("b.png", 20)]
    with patch("app.api.routes.pose.embed_with_pose", _embed_by_image(uploads, [_rejected(85), None])), \
         patch("app.api.routes.pose.register_profile_with_poses", AsyncMock()) as store:
        response = client.post(
            "/api/v1/pose/register", headers=auth_headers,
            data={"face_name": "Nobody"}, files=uploads,
        )

    assert response.status_code == 422
    assert len(response.json()["detail"]["images"]) == 2
    store.assert_not_awaited()


def test_register_rejects_too_many_files(client, auth_headers, session):
    response = client.post(
        "/api/v1/pose/register", headers=auth_headers,
        data={"face_name": "X"}, files=[_png(f"{i}.png") for i in range(11)],
    )
    assert response.status_code == 400


def test_register_requires_auth(client):
    response = client.post("/api/v1/pose/register", data={"face_name": "X"}, files=[_png("a.png")])
    assert response.status_code == 401


# ── Adding poses to an existing profile ───────────────────────────


def _face_row(session, name="Jane Doe"):
    result = MagicMock()
    result.one_or_none.return_value = MagicMock(face_name=name) if name else None
    session.execute.return_value = result


def test_add_poses_to_existing_profile(client, auth_headers, session):
    from app.core.config import settings

    _face_row(session)
    uploads = [_png("right.png", 10), _png("up.png", 20)]
    with patch("app.api.routes.pose.embed_with_pose",
               _embed_by_image(uploads, [_accepted(30), _accepted(0, 25)])), \
         patch("app.api.routes.pose.add_poses", AsyncMock(return_value=([7, 8], []))) as add, \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=1)) as audit:
        response = client.post(
            "/api/v1/pose/faces/12/poses", headers=auth_headers, files=uploads,
        )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "POSES_ADDED" and body["profileId"] == 12
    assert body["poseBuckets"] == ["frontal_up", "right_3q"]
    assert [i["poseId"] for i in body["images"]] == [7, 8]
    assert add.call_args.args[1] == 12
    assert add.call_args.kwargs["max_poses"] == settings.pose_max_per_face
    assert [c.kwargs["event_type"] for c in audit.call_args_list] == ["REGISTER_POSES"]


def test_add_poses_to_a_full_profile_drops_the_oldest(client, auth_headers, session):
    _face_row(session)
    uploads = [_png("a.png", 10), _png("b.png", 20)]
    # The database layer reports the two oldest poses it deleted to stay at the limit.
    with patch("app.api.routes.pose.embed_with_pose",
               _embed_by_image(uploads, [_accepted(30), _accepted(-30)])), \
         patch("app.api.routes.pose.add_poses", AsyncMock(return_value=([41, 42], [1, 2]))), \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=1)) as audit:
        response = client.post(
            "/api/v1/pose/faces/12/poses", headers=auth_headers, files=uploads,
        )

    # Same response as any add: the app needs no change.
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "POSES_ADDED" and body["accepted"] == 2
    assert [i["poseId"] for i in body["images"]] == [41, 42]
    assert [c.kwargs["event_type"] for c in audit.call_args_list] == [
        "REGISTER_POSES", "REGISTER_POSES_DROP_OLDEST",
    ]


def test_add_poses_unknown_face(client, auth_headers, session):
    _face_row(session, name=None)
    response = client.post("/api/v1/pose/faces/99/poses", headers=auth_headers, files=[_png("a.png")])
    assert response.status_code == 404


# ── Re-registering an existing profile ────────────────────────────


def test_replace_poses_re_registers_a_full_profile(client, auth_headers, session):
    _face_row(session)
    left, frontal = _accepted(-35, fill=0.1), _accepted(3, fill=0.3)
    left.embedding[:256] = -0.1
    uploads = [_png("left.png", 10), _png("front.png", 20), _png("blank.png", 30)]
    with patch("app.api.routes.pose.embed_with_pose", _embed_by_image(uploads, [left, frontal, None])), \
         patch("app.api.routes.pose.add_poses", AsyncMock()) as add, \
         patch("app.api.routes.pose.replace_poses", AsyncMock(return_value=[7, 8])) as replace, \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=1)) as audit:
        response = client.put(
            "/api/v1/pose/faces/12/poses", headers=auth_headers,
            data={"tenant_id": 7}, files=uploads,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "POSES_REPLACED" and body["profileId"] == 12
    assert (body["accepted"], body["rejected"]) == (2, 1)
    assert [i["poseId"] for i in body["images"]] == [7, 8, None]
    add.assert_not_awaited()

    args, kwargs = replace.call_args
    assert args[1] == 12 and kwargs["tenant_id"] == 7
    # Main embedding = normalised average of the new photos, as in /pose/register.
    expected = (left.embedding + frontal.embedding) / 2
    np.testing.assert_allclose(args[2], expected / np.linalg.norm(expected), rtol=1e-6)
    assert len(args[3]) == 2
    assert audit.call_args.kwargs["event_type"] == "REGISTER_POSES_REPLACE"


def test_replace_poses_keeps_old_poses_when_no_image_is_usable(client, auth_headers, session):
    _face_row(session)
    uploads = [_png("a.png", 10), _png("b.png", 20)]
    with patch("app.api.routes.pose.embed_with_pose", _embed_by_image(uploads, [_rejected(85), None])), \
         patch("app.api.routes.pose.replace_poses", AsyncMock()) as replace:
        response = client.put("/api/v1/pose/faces/12/poses", headers=auth_headers, files=uploads)

    assert response.status_code == 422
    replace.assert_not_awaited()


def test_replace_poses_unknown_face(client, auth_headers, session):
    _face_row(session, name=None)
    with patch("app.api.routes.pose.replace_poses", AsyncMock()) as replace:
        response = client.put("/api/v1/pose/faces/99/poses", headers=auth_headers, files=[_png("a.png")])
    assert response.status_code == 404
    replace.assert_not_awaited()


def test_replace_poses_face_deleted_meanwhile(client, auth_headers, session):
    _face_row(session)
    with patch("app.api.routes.pose.embed_with_pose", side_effect=[_accepted(0)]), \
         patch("app.api.routes.pose.replace_poses", AsyncMock(return_value=None)), \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock()) as audit:
        response = client.put("/api/v1/pose/faces/12/poses", headers=auth_headers, files=[_png("a.png")])
    assert response.status_code == 404
    audit.assert_not_awaited()


# ── Live pose estimate ────────────────────────────────────────────


def test_estimate_returns_pose_and_box_without_touching_db(client, auth_headers, session):
    from app.core.pose_pipeline import PoseEstimate

    estimate = PoseEstimate(HeadPose(28.0, -4.0, 2.0), 0.81, (0.25, 0.1, 0.6, 0.7), None)
    with patch("app.api.routes.pose.estimate_frame_pose", return_value=estimate), \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock()) as audit:
        response = client.post(
            "/api/v1/pose/estimate", headers=auth_headers,
            files={"file": ("frame.png", _png("f.png")[1][1], "image/png")},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["faceFound"] and body["usable"]
    assert body["pose"]["bucket"] == "right_3q"
    assert body["bbox"] == [0.25, 0.1, 0.6, 0.7]
    audit.assert_not_awaited()
    session.execute.assert_not_called()


def test_estimate_no_face(client, auth_headers, session):
    with patch("app.api.routes.pose.estimate_frame_pose", return_value=None):
        response = client.post(
            "/api/v1/pose/estimate", headers=auth_headers,
            files={"file": ("frame.png", _png("f.png")[1][1], "image/png")},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["faceFound"] is False and body["pose"] is None and not body["usable"]


# ── Search ───────────────────────────────────────────────────────


def _search(client, auth_headers, embed_result, matches=()):
    with patch("app.api.routes.pose.embed_with_pose", return_value=embed_result), \
         patch("app.api.routes.pose.pose_ann_query", AsyncMock(return_value=list(matches))) as query, \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=55)) as audit, \
         patch("app.api.routes.pose.create_alert", AsyncMock(return_value=66)) as alert:
        response = client.post(
            "/api/v1/pose/search", headers=auth_headers, params={"limit": 5},
            data={"tenant_id": 7, "gps_lat": 12.9, "gps_lon": 77.6},
            files={"file": ("cctv.png", _png("cctv.png")[1][1], "image/png")},
        )
    return response, query, audit, alert


def test_search_match_reports_query_and_matched_pose(client, auth_headers, session):
    from app.core.config import settings

    match = {
        "id": 41, "face_name": "Jane Doe", "alias": None, "distance": 0.27,
        "matched_pose": "left_3q", "matched_yaw": -33.0, "matched_pitch": 1.0, "tenant_id": 7,
    }
    response, query, audit, alert = _search(client, auth_headers, _accepted(-38, -12), [match])

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "MATCH"
    assert body["queryPose"]["bucket"] == "left_3q"
    assert body["matches"][0]["matchedPose"] == "left_3q"
    assert body["matchThreshold"] == settings.match_threshold_insightface
    assert body["alertId"] == 66

    assert query.call_args.kwargs == {
        "threshold": settings.match_threshold_insightface, "limit": 5, "tenant_id": 7,
    }
    assert audit.call_args.kwargs["event_type"] == "MATCH"
    assert alert.call_args.kwargs["face_id"] == 41
    assert alert.call_args.kwargs["audit_log_id"] == 55


def test_search_no_match(client, auth_headers, session):
    response, _, audit, alert = _search(client, auth_headers, _accepted(10))
    assert response.json()["status"] == "NO_MATCH"
    assert audit.call_args.kwargs["event_type"] == "NO_MATCH"
    alert.assert_not_awaited()


def test_search_low_quality_pose_is_not_searched(client, auth_headers, session):
    response, query, audit, alert = _search(client, auth_headers, _rejected(87))
    body = response.json()
    assert body["status"] == "LOW_QUALITY"
    assert body["queryPose"]["bucket"] == "right_profile"
    assert body["error"]
    query.assert_not_awaited()
    assert audit.call_args.kwargs["event_type"] == "LOW_QUALITY"
    alert.assert_not_awaited()


def test_search_no_face(client, auth_headers, session):
    response, query, _, _ = _search(client, auth_headers, None)
    body = response.json()
    assert body["status"] == "ERROR" and body["error"] == "No face detected"
    query.assert_not_awaited()
