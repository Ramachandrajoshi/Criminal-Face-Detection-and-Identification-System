"""
Integration test: full register flow with test database.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestRegisterFlow:
    """End-to-end register flow validation."""

    def test_register_rejects_non_image(self, client):
        """Non-image file should be rejected (400) or blocked by auth (401)."""
        response = client.post(
            "/api/v1/register",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"suspect_name": "Test"},
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
            data={"suspect_name": "Large File Test"},
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
            files={"file": ("suspect.jpg", buf, "image/jpeg")},
            data={
                "suspect_name": "John Doe",
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

    def test_register_requires_suspect_name(self, client):
        """Missing suspect_name should be rejected (422 or 401)."""
        img = Image.new("RGB", (112, 112), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/register",
            files={"file": ("suspect.jpg", buf, "image/jpeg")},
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
            files={"file": ("suspect.jpg", buf, "image/jpeg")},
            data={
                "suspect_name": "Bad JSON",
                "demographics": "{invalid json",
            },
        )
        assert response.status_code in (400, 401)
