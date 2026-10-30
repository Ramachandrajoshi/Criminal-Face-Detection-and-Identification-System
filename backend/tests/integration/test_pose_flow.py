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
    # Uploads for enrollment pick the centred face, under the pose version.
    assert all(call.args[1:] == (True, 4) for call in stub.call_args_list)
    assert kwargs["embedding_version"] == 4
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
    assert add.call_args.kwargs["embedding_version"] == 4
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
    assert args[1] == 12 and kwargs["tenant_id"] == 7 and kwargs["embedding_version"] == 4
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
        "threshold": settings.match_threshold_insightface, "limit": 5, "tenant_id": 7, "embedding_version": 4,
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


# ── Version 5 and tenants with poses of both versions ─────────────


def _by_version(results: dict) -> MagicMock:
    """Stub for embed_with_pose returning ``results[version]``; an exception is raised."""
    def embed(content, enrollment=False, version=4, enforce_limits=True):
        result = results[version]
        if isinstance(result, Exception):
            raise result
        return result
    return MagicMock(side_effect=embed)


def test_register_add_and_replace_under_version_5(client, auth_headers, session):
    _face_row(session)
    uploads = [_png("front.png", 10)]
    stub = _embed_by_image(uploads, [_accepted(2)])
    with patch("app.api.routes.pose.POSE_EMBEDDING_VERSION", 5), \
         patch("app.api.routes.pose.embed_with_pose", stub), \
         patch("app.api.routes.pose.register_profile_with_poses", AsyncMock(return_value=(41, [100]))) as store, \
         patch("app.api.routes.pose.add_poses", AsyncMock(return_value=([101], []))) as add, \
         patch("app.api.routes.pose.replace_poses", AsyncMock(return_value=[102])) as replace, \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=1)):
        assert client.post("/api/v1/pose/register", headers=auth_headers,
                           data={"face_name": "Jane"}, files=uploads).status_code == 201
        assert client.post("/api/v1/pose/faces/41/poses", headers=auth_headers, files=uploads).status_code == 201
        assert client.put("/api/v1/pose/faces/41/poses", headers=auth_headers, files=uploads).status_code == 200

    assert [c.args[1:] for c in stub.call_args_list] == [(True, 5)] * 3
    for db_call in (store, add, replace):
        assert db_call.call_args.kwargs["embedding_version"] == 5


def _mixed_search(client, auth_headers, results, present, matches_by_version=None):
    """POST /pose/search with POSE_EMBEDDING_VERSION=5 and ``present`` versions holding poses."""
    matches_by_version = matches_by_version or {}
    with patch("app.api.routes.pose.POSE_EMBEDDING_VERSION", 5), \
         patch("app.api.routes.pose.get_present_pose_versions", AsyncMock(return_value=set(present))), \
         patch("app.api.routes.pose.embed_with_pose", _by_version(results)) as embed, \
         patch("app.api.routes.pose.pose_ann_query", AsyncMock(
             side_effect=lambda s, e, **kw: [dict(m) for m in matches_by_version.get(kw["embedding_version"], [])]
         )) as query, \
         patch("app.api.routes.pose.add_audit_entry", AsyncMock(return_value=55)) as audit, \
         patch("app.api.routes.pose.create_alert", AsyncMock(return_value=66)):
        response = client.post(
            "/api/v1/pose/search", headers=auth_headers, params={"limit": 5}, data={"tenant_id": 7},
            files={"file": ("cctv.png", _png("cctv.png")[1][1], "image/png")},
        )
    assert response.status_code == 200, response.text
    return response.json(), embed, query, audit


def _match(face_id, distance):
    return {"id": face_id, "face_name": f"P{face_id}", "alias": None, "distance": distance,
            "matched_pose": "frontal", "matched_yaw": 0.0, "matched_pitch": 0.0, "tenant_id": 7}


def test_search_compares_each_version_with_its_own_poses_and_merges(client, auth_headers, session):
    from app.core.config import settings
    from app.db.vector_ops import compute_query_hash

    v5, v4 = _accepted(-25, fill=0.2), _accepted(-5, fill=0.1)
    with patch.object(settings, "match_threshold_insightface", 0.5):
        body, embed, query, audit = _mixed_search(
            client, auth_headers, {5: v5, 4: v4}, present={4, 5},
            matches_by_version={
                5: [_match(41, 0.30), _match(42, 0.50)],
                # 41 has poses of both versions: 0.28 / 0.5 is worse than 0.30 / 0.6.
                4: [_match(41, 0.28), _match(43, 0.20)],
            },
        )

    assert body["status"] == "MATCH"
    assert [(m["id"], m["distance"]) for m in body["matches"]] == [(43, 0.20), (41, 0.30), (42, 0.50)]
    # Pose, threshold and hash come from the configured version (5).
    assert body["queryPose"]["bucket"] == "left_3q"
    assert body["matchThreshold"] == settings.match_threshold_adaface
    assert audit.call_args.kwargs["query_hash"] == compute_query_hash(v5.embedding.tobytes())
    assert sorted(c.args[1:] for c in embed.call_args_list) == [(False, 4, False), (False, 5, False)]
    assert [(c.kwargs["embedding_version"], c.kwargs["threshold"]) for c in query.call_args_list] == [
        (5, settings.match_threshold_adaface), (4, 0.5),
    ]
    assert all(c.args[1] is {5: v5, 4: v4}[c.kwargs["embedding_version"]].embedding for c in query.call_args_list)


def test_search_only_embeds_versions_with_poses(client, auth_headers, session):
    body, embed, query, _ = _mixed_search(client, auth_headers, {5: _accepted(0)}, present=set())
    assert body["status"] == "NO_MATCH"
    assert [c.args[2] for c in embed.call_args_list] == [5]
    assert [c.kwargs["embedding_version"] for c in query.call_args_list] == [5]


def test_search_uses_the_other_version_when_one_finds_no_face(client, auth_headers, session):
    body, _, query, _ = _mixed_search(
        client, auth_headers, {5: None, 4: _accepted(30)}, present={4},
        matches_by_version={4: [_match(41, 0.31)]},
    )
    assert body["status"] == "MATCH" and body["matches"][0]["id"] == 41
    assert body["queryPose"]["bucket"] == "right_3q"
    assert [c.kwargs["embedding_version"] for c in query.call_args_list] == [4]


def test_search_pose_limits_are_decided_by_the_configured_version(client, auth_headers, session):
    body, _, query, audit = _mixed_search(
        client, auth_headers, {5: _rejected(85), 4: _accepted(70)}, present={4, 5},
    )
    assert body["status"] == "LOW_QUALITY" and body["queryPose"]["bucket"] == "right_profile"
    query.assert_not_awaited()
    assert audit.call_args.kwargs["event_type"] == "LOW_QUALITY"


@pytest.mark.parametrize("v5_matches, status", [([], "ERROR"), ([_match(41, 0.2)], "MATCH")])
def test_search_with_a_version_that_failed(client, auth_headers, session, v5_matches, status):
    """Poses of a version that could not be embedded were not compared: not a clean NO_MATCH."""
    body, _, query, _ = _mixed_search(
        client, auth_headers, {5: _accepted(0), 4: RuntimeError("CUDA error")}, present={4, 5},
        matches_by_version={5: v5_matches},
    )
    assert body["status"] == status
    if status == "ERROR":
        assert body["error"] == "Embedding extraction failed"
    assert [c.kwargs["embedding_version"] for c in query.call_args_list] == [5]


@pytest.mark.parametrize("results, error", [
    ({5: None, 4: None}, "No face detected"),
    ({5: RuntimeError("missing weights"), 4: None}, "Embedding extraction failed"),
])
def test_search_without_a_face_in_any_version(client, auth_headers, session, results, error):
    body, _, query, _ = _mixed_search(client, auth_headers, results, present={4})
    assert body["status"] == "ERROR" and body["error"] == error and body["queryPose"] is None
    query.assert_not_awaited()


def test_search_debug_dump_lists_nearest_poses_of_every_version(client, auth_headers, session):
    import contextlib

    from app.core.config import settings

    notes = MagicMock()
    with patch.object(settings, "debug_face_dump", True), \
         patch("app.api.routes.pose.face_dump", lambda *a: contextlib.nullcontext()), \
         patch("app.api.routes.pose.dump_note", notes):
        body, _, query, _ = _mixed_search(
            client, auth_headers, {5: _accepted(0), 4: _accepted(0)}, present={4, 5},
            matches_by_version={5: [_match(41, 0.7)], 4: [_match(42, 0.9)]},
        )

    assert body["status"] == "MATCH"   # the stub ignores the threshold
    nearest = dict((c.args[0], c.args[1]) for c in notes.call_args_list)["nearest_ignoring_threshold"]
    assert nearest == [{"version": 5, "face_id": 41, "distance": 0.7}, {"version": 4, "face_id": 42, "distance": 0.9}]
    assert {c.kwargs["threshold"] for c in query.call_args_list[2:]} == {2.0}
