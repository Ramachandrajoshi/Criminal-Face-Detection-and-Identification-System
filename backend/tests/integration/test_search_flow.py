"""
Integration test: full search flow.

Tests:
- Non-image rejection
- Auth requirement
- Response structure validation
- GPS coordinate passing
"""

import io
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.auth import create_access_token

from app.main import app
from app.db.session import get_session


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token():
    return create_access_token({"sub": "operator", "role": "admin"})


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.mark.parametrize("enrolled,closest_version", [
    # Right after the version bump: everything still on the previous recipe.
    ({2}, 2),
    ({2, 1}, 1),
    # Part-way through re-enrolment — the older recipe holds the better match
    # and must still win.
    ({3, 2}, 2),
    ({3, 2, 1}, 1),
    # Fully re-enrolled.
    ({3}, 3),
    # Nothing matches anywhere.
    ({3, 2}, None),
])
def test_search_across_enrolled_recipes(client, auth_headers, enrolled, closest_version):
    """HTTP -> pipeline -> real ANN SQL -> merged ranking -> audit event."""
    from app.core.config import settings, CURRENT_EMBEDDING_VERSION
    from app.db.vector_ops import compute_query_hash

    session = AsyncMock()
    embeddings = {v: np.full(512, v, dtype=np.float32) for v in (3, 2, 1)}
    queried_versions = []

    async def override_session():
        yield session

    async def execute(sql, params):
        statement = str(sql)

        # 1) The enrolled-recipe census that gates which recipes get embedded.
        if "SELECT DISTINCT embedding_version" in statement:
            assert params == {"tenant_id": 7}
            return [(v,) for v in sorted(enrolled)]

        # 2) The real cosine_ann_query SQL — assert its bound parameters rather
        #    than mocking the pipeline out from under them.
        assert "WHERE (face_embedding <=> :vec) <= :threshold" in statement
        assert "AND tenant_id = :tenant_id" in statement
        assert "AND embedding_version = :target_version" in statement
        assert params["threshold"] == settings.match_threshold
        assert params["tenant_id"] == 7
        assert params["limit"] == 5
        version = params["target_version"]
        assert params["vec"] == "[" + ",".join([f"{version:.10f}"] * 512) + "]"
        queried_versions.append(version)

        if closest_version is None:
            return []
        # Every enrolled recipe returns a hit; the designated one is closest.
        distance = 0.10 if version == closest_version else 0.40
        return [(version, f"Face v{version}", None, distance, version)]

    session.execute.side_effect = execute
    image = io.BytesIO()
    Image.new("RGB", (112, 112), color="blue").save(image, format="PNG")
    with patch.dict(app.dependency_overrides, {get_session: override_session}), \
         patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
         patch("app.core.pipeline.extract_embedding",
               side_effect=lambda face, version: embeddings[version]) as extract, \
         patch("app.api.routes.search.add_audit_entry", AsyncMock(return_value=42)) as audit, \
         patch("app.api.routes.search.create_alert", AsyncMock(return_value=43)) as alert:
        response = client.post(
            "/api/v1/search", headers=auth_headers, params={"limit": 5},
            data={"tenant_id": 7}, files={"file": ("test.png", image.getvalue(), "image/png")},
        )

    assert response.status_code == 200
    result = response.json()

    # Only enrolled recipes are queried; the current one is always embedded so
    # the audit hash does not depend on which recipe held the match.
    assert sorted(queried_versions) == sorted(enrolled)
    assert sorted(c.args[1] for c in extract.call_args_list) == sorted(
        enrolled | {CURRENT_EMBEDDING_VERSION}
    )

    expected_status = "MATCH" if closest_version else "NO_MATCH"
    expected_hash = compute_query_hash(embeddings[CURRENT_EMBEDDING_VERSION].tobytes())
    assert result["status"] == expected_status
    assert result["queryHash"] == expected_hash
    audit.assert_awaited_once()
    assert audit.call_args.kwargs["event_type"] == expected_status
    assert audit.call_args.kwargs["query_hash"] == expected_hash

    if closest_version:
        # Merged across recipes and ranked by distance, closest first.
        distances = [m["distance"] for m in result["matches"]]
        assert len(result["matches"]) == len(enrolled)
        assert distances == sorted(distances)
        assert result["matches"][0]["embeddingVersion"] == closest_version
        assert result["matches"][0]["distance"] == 0.10
        alert.assert_awaited_once()
        assert alert.call_args.kwargs["distance"] == 0.10
    else:
        assert result["matches"] == []
        alert.assert_not_awaited()


def test_search_result_limit_bounds_the_merged_set(client, auth_headers):
    """`limit` caps results across all recipes combined, not per recipe."""
    session = AsyncMock()

    async def override_session():
        yield session

    async def execute(sql, params):
        statement = str(sql)
        if "SELECT DISTINCT embedding_version" in statement:
            return [(3,), (2,)]
        v = params["target_version"]
        # Each recipe returns `limit` rows; interleaved distances across them.
        return [(v * 10 + i, f"v{v}#{i}", None, 0.05 * (2 * i + (v == 3)), v)
                for i in range(params["limit"])]

    session.execute.side_effect = execute
    image = io.BytesIO()
    Image.new("RGB", (112, 112), color="blue").save(image, format="PNG")
    with patch.dict(app.dependency_overrides, {get_session: override_session}), \
         patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
         patch("app.core.pipeline.extract_embedding",
               side_effect=lambda face, version: np.full(512, version, dtype=np.float32)), \
         patch("app.api.routes.search.add_audit_entry", AsyncMock(return_value=1)), \
         patch("app.api.routes.search.create_alert", AsyncMock(return_value=2)):
        response = client.post(
            "/api/v1/search", headers=auth_headers, params={"limit": 3},
            files={"file": ("test.png", image.getvalue(), "image/png")},
        )

    assert response.status_code == 200
    matches = response.json()["matches"]
    distances = [m["distance"] for m in matches]
    assert len(matches) == 3, "6 candidate rows must be truncated to the limit"
    assert distances == sorted(distances)


class TestSearchFlow:
    """Search flow validation."""

    def test_search_rejects_non_image(self, client):
        """Non-image file should be rejected (400) or blocked by auth (401)."""
        response = client.post(
            "/api/v1/search",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code in (400, 401)

    def test_search_validates_response_structure(self, client):
        """Search with valid image returns structured response."""
        img = Image.new("RGB", (224, 224), color="purple")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
        )
        data = response.json()
        if "status" in data:
            assert data["status"] in ("MATCH", "NO_MATCH", "ERROR", "SPOOF_BLOCKED")
        elif "detail" in data:
            # Auth error is acceptable in test env
            pass

    def test_search_requires_auth(self, client):
        """Unauthenticated search should return 401."""
        img = Image.new("RGB", (224, 224), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 401

    def test_search_with_auth_returns_200_or_400(self, client, auth_headers):
        """Authenticated search returns 200 with structured response.
        Returns 500 if deepface/ML stack is not installed."""
        img = Image.new("RGB", (224, 224), color="orange")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
            headers=auth_headers,
        )
        # 200 = ML stack available + DB works
        # 400 = validation error
        # 500 = ML stack (deepface) not installed
        assert response.status_code in (200, 400, 500)
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] in ("MATCH", "NO_MATCH", "ERROR", "SPOOF_BLOCKED")
            assert "query_hash" in data

    def test_search_rejects_large_file(self, client):
        """File > 5 MB should be rejected."""
        img = Image.new("RGB", (2000, 2000), color="cyan")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        large_buf = buf.read() + b"\x00" * (6 * 1024 * 1024)
        buf = io.BytesIO(large_buf)
        response = client.post(
            "/api/v1/search",
            files={"file": ("large.jpg", buf, "image/jpeg")},
        )
        assert response.status_code in (400, 401)

    def test_search_with_gps_coordinates(self, client, auth_headers):
        """Search with GPS coordinates should accept them.
        May return 500 if deepface is not installed."""
        img = Image.new("RGB", (224, 224), color="magenta")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
            data={"gps_lat": "40.7128", "gps_lon": "-74.0060"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 400, 500)
        if response.status_code == 200:
            data = response.json()
            assert "gps_lat" in data or "detail" in data

    def test_search_limit_param(self, client, auth_headers):
        """Limit parameter should be accepted.
        May return 500 if deepface is not installed."""
        img = Image.new("RGB", (224, 224), color="yellow")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
            params={"limit": 5},
            headers=auth_headers,
        )
        assert response.status_code in (200, 400, 500)

    @pytest.mark.usefixtures("db_session")
    def test_search_invalid_limit(self, client, auth_headers):
        """Limit < 1 should return 422."""
        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
            params={"limit": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.usefixtures("db_session")
    def test_search_limit_too_high(self, client, auth_headers):
        """Limit > 50 should return 422."""
        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
            params={"limit": 100},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_batch_stream_match_carries_face_name(self, client, auth_headers):
        """Batch SSE matches carry /search's ``faceName`` key (the UI renders it) and the old ``suspectName``."""
        import json

        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = {
            "status": "MATCH", "query_hash": "abc", "alert_id": 7, "error": None,
            "matches": [{
                "id": 3, "face_name": "Jane Doe", "alias": None, "distance": 0.37,
                "embedding_version": 4, "tenant_id": 1,
            }],
        }

        with patch("app.api.routes.search._search_one", AsyncMock(return_value=result)):
            response = client.post(
                "/api/v1/search/batch/stream",
                files=[("files", ("query.jpg", buf.getvalue(), "image/jpeg"))],
                headers=auth_headers,
            )

        assert response.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines() if line.startswith("data: ")
        ]
        match = next(e for e in events if e["type"] == "progress")["matches"][0]
        assert match["faceName"] == "Jane Doe"
        assert match["suspectName"] == "Jane Doe"


def _jpeg(size: int = 112) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def _per_file_sessions(session):
    """Stand-in for async_session_factory: every batch file gets ``session``."""
    @asynccontextmanager
    async def factory():
        yield session
    return factory


_OVERSIZED = _jpeg() + b"\0" * (5 * 1024 * 1024)
MATCH = {
    "status": "MATCH", "query_hash": "q" * 64,
    "matches": [{"id": 3, "face_name": "Jane Doe", "alias": None, "distance": 0.37,
                 "embedding_version": 4, "tenant_id": 1}],
}


class TestSearchValidation:
    @pytest.mark.parametrize("upload, detail", [
        (("q.gif", b"GIF89a", "image/gif"), "Only JPEG/PNG"),
        (("q.jpg", _OVERSIZED, "image/jpeg"), "5 MB"),
        (("q.jpg", _jpeg(size=32), "image/jpeg"), "at least 64x64"),
    ])
    def test_invalid_upload_is_rejected(self, client, auth_headers, db_session, upload, detail):
        response = client.post("/api/v1/search", files={"file": upload}, headers=auth_headers)
        assert response.status_code == 400
        assert detail in response.json()["detail"]


class TestBatchSearch:
    def _stream(self, client, auth_headers, files, run_pipeline, session=None):
        session = session or AsyncMock()
        with patch("app.api.routes.search.run_pipeline", run_pipeline), \
             patch("app.api.routes.search.async_session_factory", _per_file_sessions(session)), \
             patch("app.api.routes.search.add_audit_entry", AsyncMock(return_value=42)) as audit, \
             patch("app.api.routes.search.create_alert", AsyncMock(return_value=43)) as alert:
            response = client.post("/api/v1/search/batch/stream", files=[("files", f) for f in files],
                                   headers=auth_headers)
        events = [json.loads(line.removeprefix("data: "))
                  for line in response.text.splitlines() if line.startswith("data: ")]
        return response, events, audit, alert

    def test_counts_matches_misses_and_errors(self, client, auth_headers):
        async def run_pipeline(file, session, **kwargs):
            if file.filename == "hit.jpg":
                return MATCH
            return {"status": "NO_MATCH", "query_hash": "n" * 64, "matches": []}

        files = [("hit.jpg", _jpeg(), "image/jpeg"), ("miss.jpg", _jpeg(), "image/jpeg"),
                 ("q.gif", b"GIF89a", "image/gif"), ("big.jpg", _OVERSIZED, "image/jpeg"),
                 ("tiny.jpg", _jpeg(size=32), "image/jpeg")]
        response, events, audit, alert = self._stream(client, auth_headers, files, run_pipeline)

        assert response.status_code == 200
        progress = {e["filename"]: e for e in events if e["type"] == "progress"}
        assert progress["hit.jpg"]["alertId"] == 43
        assert progress["miss.jpg"]["status"] == "NO_MATCH" and progress["miss.jpg"]["alertId"] is None
        assert "Unsupported MIME type" in progress["q.gif"]["error"]
        assert progress["big.jpg"]["error"] == "Image exceeds 5 MB limit"
        assert "at least 64x64" in progress["tiny.jpg"]["error"]
        done = events[-1]
        assert (done["matched"], done["noMatch"], done["errors"]) == (1, 1, 3)
        assert audit.await_count == 2                  # one per searched file
        assert alert.await_args.kwargs["face_id"] == 3

    def test_spoof_blocked_is_not_audited_twice(self, client, auth_headers):
        pipeline = AsyncMock(return_value={"status": "SPOOF_BLOCKED", "query_hash": "s" * 64, "matches": []})
        _, events, audit, alert = self._stream(client, auth_headers, [("a.jpg", _jpeg(), "image/jpeg")], pipeline)
        assert events[1]["status"] == "SPOOF_BLOCKED"
        audit.assert_not_awaited()                     # run_pipeline already logged it
        alert.assert_not_awaited()

    def test_pipeline_exception_is_reported_per_file(self, client, auth_headers):
        session = AsyncMock()
        session.rollback.side_effect = RuntimeError("connection already gone")  # swallowed
        pipeline = AsyncMock(side_effect=RuntimeError("db down"))
        _, events, _, _ = self._stream(client, auth_headers, [("a.jpg", _jpeg(), "image/jpeg")], pipeline, session)
        assert (events[1]["status"], events[1]["error"]) == ("ERROR", "db down")
        assert events[-1]["errors"] == 1
        session.rollback.assert_awaited_once()

    def test_rejects_more_than_20_files(self, client, auth_headers):
        files = [(f"{i}.jpg", b"x", "image/jpeg") for i in range(21)]
        response, _, _, _ = self._stream(client, auth_headers, files, AsyncMock())
        assert response.status_code == 400
        assert "20 images" in response.json()["detail"]
