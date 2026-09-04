"""
Integration test: standalone liveness check endpoint.

Tests:
- Auth requirement
- Non-image rejection
- Oversized file rejection
- Response structure validation
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


class TestLivenessFlow:
    """Liveness endpoint validation."""

    def test_liveness_requires_auth(self, client):
        """Unauthenticated request should return 401."""
        img = Image.new("RGB", (224, 224), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/liveness",
            files={"file": ("frame.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 401

    def test_liveness_rejects_non_image(self, client, auth_headers):
        """Non-image file should be rejected (400)."""
        response = client.post(
            "/api/v1/liveness",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_liveness_rejects_large_file(self, client, auth_headers):
        """File > 5 MB should be rejected."""
        img = Image.new("RGB", (2000, 2000), color="cyan")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        large_buf = buf.getvalue() + b"\x00" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/liveness",
            files={"file": ("large.jpg", io.BytesIO(large_buf), "image/jpeg")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_liveness_with_auth_returns_200_or_500(self, client, auth_headers):
        """Authenticated request returns a structured liveness verdict.
        Returns 500 if the deepface/ML stack is not installed."""
        img = Image.new("RGB", (224, 224), color="orange")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/liveness",
            files={"file": ("frame.jpg", buf, "image/jpeg")},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            data = response.json()
            assert "isLive" in data
            assert isinstance(data["isLive"], bool)
            assert "spoofProbability" in data
            assert 0.0 <= data["spoofProbability"] <= 1.0
            assert "message" in data

    def test_liveness_does_not_write_audit_entry(self, client, auth_headers, monkeypatch):
        """The standalone liveness check must not touch the audit log."""
        from app.db import vector_ops

        called = []

        async def _fail_if_called(*args, **kwargs):
            called.append((args, kwargs))

        monkeypatch.setattr(vector_ops, "add_audit_entry", _fail_if_called)

        img = Image.new("RGB", (224, 224), color="orange")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        client.post(
            "/api/v1/liveness",
            files={"file": ("frame.jpg", buf, "image/jpeg")},
            headers=auth_headers,
        )
        assert called == []


class TestVideoLivenessFlow:
    """Video liveness endpoint validation."""

    def test_video_liveness_requires_auth(self, client):
        response = client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        )
        assert response.status_code == 401

    def test_video_liveness_rejects_bad_mime(self, client, auth_headers):
        response = client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.txt", b"not a video", "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_video_liveness_rejects_large_file(self, client, auth_headers):
        large = b"\x00" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.mp4", large, "video/mp4")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_video_liveness_returns_400_for_bad_duration(self, client, auth_headers, monkeypatch):
        """Route should translate a duration_out_of_range verdict into 400."""
        def fake_analyze(video_bytes, **kwargs):
            return {
                "is_live": False, "spoof_probability": 1.0,
                "frames_analyzed": 0, "duration_sec": 1.0,
                "error": "duration_out_of_range",
            }

        monkeypatch.setattr("app.api.routes.liveness.analyze_video_liveness", fake_analyze)

        response = client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "duration" in response.json()["detail"].lower()

    def test_video_liveness_returns_400_for_invalid_video(self, client, auth_headers, monkeypatch):
        def fake_analyze(video_bytes, **kwargs):
            return {
                "is_live": False, "spoof_probability": 1.0,
                "frames_analyzed": 0, "duration_sec": 0.0,
                "error": "invalid_video",
            }

        monkeypatch.setattr("app.api.routes.liveness.analyze_video_liveness", fake_analyze)

        response = client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_video_liveness_returns_structured_verdict(self, client, auth_headers, monkeypatch):
        """Route should surface a normal verdict as a 200 with the expected shape."""
        def fake_analyze(video_bytes, **kwargs):
            return {
                "is_live": True, "spoof_probability": 0.12,
                "frames_analyzed": 6, "duration_sec": 4.0,
            }

        monkeypatch.setattr("app.api.routes.liveness.analyze_video_liveness", fake_analyze)

        response = client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.webm", b"fake video bytes", "video/webm")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["isLive"] is True
        assert data["spoofProbability"] == 0.12
        assert data["framesAnalyzed"] == 6
        assert "message" in data

    def test_video_liveness_does_not_write_audit_entry(self, client, auth_headers, monkeypatch):
        from app.db import vector_ops

        called = []

        async def _fail_if_called(*args, **kwargs):
            called.append((args, kwargs))

        monkeypatch.setattr(vector_ops, "add_audit_entry", _fail_if_called)
        monkeypatch.setattr(
            "app.api.routes.liveness.analyze_video_liveness",
            lambda video_bytes, **kwargs: {
                "is_live": True, "spoof_probability": 0.1,
                "frames_analyzed": 6, "duration_sec": 4.0,
            },
        )

        client.post(
            "/api/v1/liveness/video",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
            headers=auth_headers,
        )
        assert called == []
