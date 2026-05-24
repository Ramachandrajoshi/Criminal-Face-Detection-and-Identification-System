"""
Test JWT authentication — create token, decode, expiry, invalid tokens.
"""

import io
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from app.core.auth import (
    create_access_token,
    decode_access_token,
    get_current_user,
)
from app.core.config import settings


class TestCreateAccessToken:
    """Test JWT token creation."""

    def test_token_contains_exp(self):
        token = create_access_token({"sub": "user123"})
        payload = decode_access_token(token)
        assert "exp" in payload
        assert payload["sub"] == "user123"

    def test_token_expiry_defaults_to_config(self):
        """Token expiry should match jwt_expiry_hours from config."""
        token = create_access_token({"sub": "user123"})
        payload = decode_access_token(token)
        expire = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiry_hours)
        # Allow 1 second tolerance for execution time
        diff = abs((expire - expected).total_seconds())
        assert diff < 2

    def test_token_custom_expiry(self):
        """Custom expires_delta should override config."""
        token = create_access_token({"sub": "user123"}, expires_delta=timedelta(hours=1))
        payload = decode_access_token(token)
        expire = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected = datetime.now(timezone.utc) + timedelta(hours=1)
        diff = abs((expire - expected).total_seconds())
        assert diff < 2

    def test_token_is_string(self):
        token = create_access_token({"sub": "user123"})
        assert isinstance(token, str)
        assert "." in token  # JWT format: header.payload.signature


class TestDecodeAccessToken:
    """Test JWT token decoding and validation."""

    def test_decode_valid_token(self):
        token = create_access_token({"sub": "user456", "role": "admin"})
        payload = decode_access_token(token)
        assert payload["sub"] == "user456"
        assert payload["role"] == "admin"

    def test_decode_invalid_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("invalid.token.here")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_empty_string_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_expired_token_raises_401(self):
        """Token with expired expiry should be rejected."""
        token = create_access_token(
            {"sub": "expired"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_expired_token_with_verify_exp_false(self):
        """Token with expired expiry should decode successfully if verify_exp is False."""
        token = create_access_token(
            {"sub": "expired-refreshed", "role": "admin"},
            expires_delta=timedelta(seconds=-1),
        )
        payload = decode_access_token(token, verify_exp=False)
        assert payload["sub"] == "expired-refreshed"
        assert payload["role"] == "admin"


class TestAuthMiddleware:
    """Test AuthMiddleware — protected routes require JWT, health does not."""

    def test_health_no_auth_required(self):
        """Health endpoint should return 200 without any auth header."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_register_requires_auth(self):
        """Register without auth should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        import io
        from PIL import Image
        img = Image.new("RGB", (112, 112), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/register",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={"suspect_name": "No Auth"},
        )
        assert response.status_code == 401

    def test_search_requires_auth(self):
        """Search without auth should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        img = Image.new("RGB", (112, 112), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/search",
            files={"file": ("query.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 401

    def test_alerts_requires_auth(self):
        """Alerts without auth should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.get("/api/v1/alerts")
        assert response.status_code == 401

    def test_audit_requires_auth(self):
        """Audit without auth should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.get("/api/v1/audit")
        assert response.status_code == 401

    def test_invalid_bearer_token_returns_401(self):
        """Request with invalid Bearer token should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.get(
            "/api/v1/alerts",
            headers={"Authorization": "Bearer invalid.token.value"},
        )
        assert response.status_code == 401

    def test_missing_bearer_prefix_returns_401(self):
        """Request without 'Bearer ' prefix should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.get(
            "/api/v1/alerts",
            headers={"Authorization": "Token abc123"},
        )
        assert response.status_code == 401
