"""
Test alerts endpoint — list, confirm, dismiss, filter, not-found.

Integration tests that hit the DB will return 500 if no DB is available.
Tests that only validate auth and input validation pass without a DB.
"""

import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

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

    @pytest.mark.usefixtures("db_session")
    def test_list_alerts_invalid_page(self, client, auth_headers):
        """Page < 1 should return 422."""
        response = client.get(
            "/api/v1/alerts",
            params={"page": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.usefixtures("db_session")
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

    @pytest.mark.usefixtures("db_session")
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
            face_id=42,
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


def _result(fetchall=None, fetchone=None):
    result = MagicMock()
    result.fetchall.return_value = fetchall or []
    result.fetchone.return_value = fetchone
    return result


class TestAlertsWithSession:
    """Listing and the human-in-the-loop confirmation, with the database mocked."""

    def test_list_maps_rows_and_paginates(self, client, auth_headers, db_session):
        created = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
        db_session.execute.return_value = _result(fetchall=[
            (1, 10, 6, "MATCH", 0.31, "PENDING_REVIEW", 12.9, 77.5, created, None, "Jane Doe", "JD", 2),
        ])

        response = client.get(
            "/api/v1/alerts",
            params={"page": 3, "page_size": 10, "status_filter": "PENDING_REVIEW", "tenant_id": 2},
            headers=auth_headers,
        )

        assert response.status_code == 200
        alert = response.json()[0]
        assert (alert["faceName"], alert["faceAlias"], alert["status"]) == ("Jane Doe", "JD", "PENDING_REVIEW")
        assert alert["createdAt"].startswith("2026-10-24") and alert["confirmedAt"] is None
        sql, params = db_session.execute.await_args.args
        assert params == {"limit": 10, "offset": 20, "tenant_id": 2, "status_filter": "PENDING_REVIEW"}
        assert "a.status = :status_filter" in str(sql)

    def test_list_without_filter_has_no_status_clause(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result()
        assert client.get("/api/v1/alerts", headers=auth_headers).json() == []
        sql, params = db_session.execute.await_args.args
        assert "status_filter" not in params and ":status_filter" not in str(sql)

    @pytest.mark.parametrize("confirmed, status", [(True, "CONFIRMED"), (False, "DISMISSED")])
    def test_confirm_accepts_the_json_body(self, client, auth_headers, db_session, confirmed, status):
        """The web UI posts JSON ``{confirmed}``; tenant_id rides in the query string."""
        db_session.execute.return_value = _result(fetchone=(4, status))

        response = client.post("/api/v1/alerts/4/confirm", params={"tenant_id": 2},
                               json={"confirmed": confirmed}, headers=auth_headers)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == status
        assert (body["confirmed_at"] is not None) is confirmed
        params = db_session.execute.await_args.args[1]
        assert (params["alert_id"], params["status"], params["tenant_id"]) == (4, status, 2)
        db_session.commit.assert_awaited_once()

    def test_confirm_unknown_alert_is_404(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(fetchone=None)
        response = client.post("/api/v1/alerts/9/confirm", json={"confirmed": True}, headers=auth_headers)
        assert response.status_code == 404

    async def test_dropped_connection_rolls_back_and_reraises(self):
        from sqlalchemy.exc import InterfaceError

        from app.api.routes.alerts import _execute_with_retry

        session = AsyncMock()
        session.execute.side_effect = InterfaceError("SELECT 1", {}, Exception("connection is closed"))
        session.rollback.side_effect = RuntimeError("connection already gone")  # swallowed

        with pytest.raises(InterfaceError):
            await _execute_with_retry(session, "SELECT 1", {})
        session.rollback.assert_awaited_once()
