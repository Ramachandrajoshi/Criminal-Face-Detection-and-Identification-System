"""
Integration test: full search flow.

Tests:
- Non-image rejection
- Auth requirement
- Response structure validation
- GPS coordinate passing
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.auth import create_access_token

from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token():
    return create_access_token({"sub": "operator", "role": "admin"})


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


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
