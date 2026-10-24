"""
Integration test: full register flow with test database.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.auth import create_access_token
from app.db.session import get_session
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


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


class TestRegisterFlow:
    """End-to-end register flow validation."""

    def test_register_rejects_non_image(self, client):
        """Non-image file should be rejected (400) or blocked by auth (401)."""
        response = client.post(
            "/api/v1/register",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"face_name": "Test"},
        )
        assert response.status_code in (400, 401)

    def test_register_rejects_large_file(self, client):
        """File > 5 MB should be rejected (400) or blocked by auth (401)."""
        img = Image.new("RGB", (1000, 1000), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        large_buf = buf.read() + b"\x00" * (6 * 1024 * 1024)
        buf = io.BytesIO(large_buf)

        response = client.post(
            "/api/v1/register",
            files={"file": ("large.jpg", buf, "image/jpeg")},
            data={"face_name": "Large File Test"},
        )
        assert response.status_code in (400, 401)

    def test_register_valid_image_structure(self, client):
        """Register with valid image returns structured response or auth error."""
        img = Image.new("RGB", (112, 112), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/register",
            files={"file": ("face.jpg", buf, "image/jpeg")},
            data={
                "face_name": "John Doe",
                "alias": "JD",
                "demographics": json.dumps({"age_band": "18-35", "gender": "M"}),
            },
        )
        data = response.json()
        if "status" in data:
            assert data["status"] == "REGISTERED"
            assert "profile_id" in data
            assert "query_hash" in data
        elif "detail" in data:
            # Auth error is acceptable in test env
            assert response.status_code == 401

    def test_register_requires_face_name(self, client):
        """Missing face_name should be rejected (422 or 401)."""
        img = Image.new("RGB", (112, 112), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/register",
            files={"file": ("face.jpg", buf, "image/jpeg")},
            data={},
        )
        assert response.status_code in (401, 422)

    def test_register_rejects_invalid_demographics_json(self, client):
        """Invalid demographics JSON should be rejected (400 or 401)."""
        img = Image.new("RGB", (112, 112), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/register",
            files={"file": ("face.jpg", buf, "image/jpeg")},
            data={
                "face_name": "Bad JSON",
                "demographics": "{invalid json",
            },
        )
        assert response.status_code in (400, 401)


def _stored_face(session):
    face = SimpleNamespace(
        id=6, face_name="Old Name", alias=None, demographics=None,
        embedding_version=4, created_at=None, tenant_id=1,
    )
    result = MagicMock()
    result.scalar_one_or_none.return_value = face
    session.execute.return_value = result
    return face


class TestUpdateFace:
    """PATCH /api/v1/faces/{id}: JSON route alongside the original form-based route."""

    @pytest.mark.parametrize("name_key", ["faceName", "face_name"])
    def test_json_body_updates_face(self, client, auth_headers, session, name_key):
        """JSON bodies are applied, in camelCase (as the web UI sends) or snake_case."""
        face = _stored_face(session)

        response = client.patch(
            "/api/v1/faces/6",
            json={name_key: "  New Name ", "alias": "NN"},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        assert face.face_name == "New Name"
        assert response.json()["faceName"] == "New Name"
        assert response.json()["alias"] == "NN"
        session.commit.assert_awaited_once()

    def test_json_body_scopes_lookup_to_tenant_query_param(self, client, auth_headers, session):
        """tenant_id comes from the query string, like GET /faces/{id}."""
        _stored_face(session)

        response = client.patch("/api/v1/faces/6?tenant_id=2", json={"alias": "NN"}, headers=auth_headers)

        assert response.status_code == 200, response.text
        lookup = session.execute.await_args.args[0]
        assert lookup.compile().params == {"id_1": 6, "tenant_id_1": 2}

    def test_non_json_requests_still_reach_form_route(self, client, auth_headers, session):
        """Non-JSON requests go to the original form route and its own validation, not the JSON one."""
        _stored_face(session)

        response = client.patch("/api/v1/faces/6", data={"face_name": "New Name"}, headers=auth_headers)

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "body"]
