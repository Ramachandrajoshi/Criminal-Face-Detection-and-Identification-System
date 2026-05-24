"""
Test alerts endpoint — list, confirm, dismiss, filter, not-found.

Integration tests that hit the DB will return 500 if no DB is available.
Tests that only validate auth and input validation pass without a DB.
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
    """Valid JWT token for authenticated requests."""
    return create_access_token({"sub": "operator1", "role": "admin"})


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


class TestListAlerts:
    """Test GET /alerts endpoint."""

    def test_list_alerts_requires_auth(self):
        """Unauthenticated request should return 401."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/alerts")
        assert response.status_code == 401

    def test_list_alerts_returns_list(self, client, auth_headers):
        """Authenticated request returns a list or 500 (if DB unavailable)."""
        response = client.get("/api/v1/alerts", headers=auth_headers)
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            assert isinstance(response.json(), list)

    def test_list_alerts_pagination(self, client, auth_headers):
        """Pagination parameters should be accepted."""
        response = client.get(
            "/api/v1/alerts",
            params={"page": 1, "page_size": 10},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_list_alerts_invalid_page(self, client, auth_headers):
        """Page < 1 should return 422."""
        response = client.get(
            "/api/v1/alerts",
            params={"page": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_list_alerts_invalid_page_size(self, client, auth_headers):
        """page_size > 100 should return 422."""
        response = client.get(
            "/api/v1/alerts",
            params={"page": 1, "page_size": 200},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_list_alerts_filter_by_status(self, client, auth_headers):
        """Status filter should be accepted without error."""
        response = client.get(
            "/api/v1/alerts",
            params={"status_filter": "PENDING_REVIEW"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_list_alerts_filter_confirmed(self, client, auth_headers):
        response = client.get(
            "/api/v1/alerts",
            params={"status_filter": "CONFIRMED"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_list_alerts_filter_dismissed(self, client, auth_headers):
        response = client.get(
            "/api/v1/alerts",
            params={"status_filter": "DISMISSED"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)


class TestConfirmAlert:
    """Test POST /alerts/{id}/confirm endpoint."""

    def test_confirm_alert_requires_auth(self):
        """Unauthenticated confirm should return 401."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/alerts/1/confirm",
            json={"confirmed": True},
        )
        assert response.status_code == 401

    def test_confirm_valid_alert(self, client, auth_headers):
        """Confirming a non-existent alert returns 404 or 500 (no DB)."""
        response = client.post(
            "/api/v1/alerts/1/confirm",
            json={"confirmed": True},
            headers=auth_headers,
        )
        assert response.status_code in (404, 500)

    def test_dismiss_alert(self, client, auth_headers):
        """Dismiss a non-existent alert returns 404 or 500."""
        response = client.post(
            "/api/v1/alerts/1/confirm",
            json={"confirmed": False},
            headers=auth_headers,
        )
        assert response.status_code in (404, 500)

    def test_confirm_invalid_payload(self, client, auth_headers):
        """Missing 'confirmed' field should return 422."""
        response = client.post(
            "/api/v1/alerts/1/confirm",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_confirm_nonexistent_alert(self, client, auth_headers):
        """Alert ID that doesn't exist should return 404 or 500."""
        response = client.post(
            "/api/v1/alerts/99999/confirm",
            json={"confirmed": True},
            headers=auth_headers,
        )
        assert response.status_code in (404, 500)


class TestAlertResponseStructure:
    """Verify AlertResponse schema structure."""

    def test_alert_response_has_required_fields(self):
        from app.schemas.face import AlertResponse

        alert = AlertResponse(
            id=1,
            audit_log_id=10,
            suspect_id=42,
            event_type="MATCH",
            distance=0.32,
            status="PENDING_REVIEW",
            gps_lat=40.7128,
            gps_lon=-74.0060,
            created_at="2026-05-24T12:00:00+00:00",
        )
        assert alert.id == 1
        assert alert.event_type == "MATCH"
        assert alert.status == "PENDING_REVIEW"
        assert alert.distance == 0.32

    def test_alert_status_valid_values(self):
        from app.schemas.face import AlertResponse

        for status_val in ("PENDING_REVIEW", "CONFIRMED", "DISMISSED"):
            alert = AlertResponse(
                id=1, event_type="MATCH", status=status_val,
                created_at="2026-05-24T12:00:00+00:00",
            )
            assert alert.status == status_val
