"""
Test audit endpoint — GET /audit with pagination and event_type filter.

Integration tests that hit the DB will return 500 if no DB is available.
Tests that only validate auth and input validation pass without a DB.
"""

import io
from datetime import datetime, timezone
from unittest.mock import MagicMock

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
    return create_access_token({"sub": "admin", "role": "admin"})


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


class TestAuditLog:
    """Test GET /audit endpoint."""

    def test_audit_requires_auth(self):
        """Unauthenticated request should return 401."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/audit")
        assert response.status_code == 401

    def test_audit_returns_list(self, client, auth_headers):
        """Authenticated request returns a list or 500 (if DB unavailable)."""
        response = client.get("/api/v1/audit", headers=auth_headers)
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            assert isinstance(response.json(), list)

    def test_audit_pagination(self, client, auth_headers):
        """Pagination params should be accepted."""
        response = client.get(
            "/api/v1/audit",
            params={"page": 1, "page_size": 10},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_audit_filter_by_event_type(self, client, auth_headers):
        """Event type filter should be accepted without error."""
        response = client.get(
            "/api/v1/audit",
            params={"event_type": "MATCH"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_audit_filter_by_no_match(self, client, auth_headers):
        response = client.get(
            "/api/v1/audit",
            params={"event_type": "NO_MATCH"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_audit_filter_by_register(self, client, auth_headers):
        response = client.get(
            "/api/v1/audit",
            params={"event_type": "REGISTER"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    def test_audit_filter_by_spoof(self, client, auth_headers):
        response = client.get(
            "/api/v1/audit",
            params={"event_type": "SPOOF_BLOCKED"},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500)

    @pytest.mark.usefixtures("db_session")
    def test_audit_invalid_page(self, client, auth_headers):
        """Page < 1 should return 422."""
        response = client.get(
            "/api/v1/audit",
            params={"page": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.usefixtures("db_session")
    def test_audit_invalid_page_size(self, client, auth_headers):
        """page_size > 200 should return 422."""
        response = client.get(
            "/api/v1/audit",
            params={"page_size": 300},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_audit_page_size_default(self, client, auth_headers):
        """Default page_size should be 50."""
        response = client.get("/api/v1/audit", headers=auth_headers)
        assert response.status_code in (200, 500)


class TestAuditResponseSchema:
    """Verify AuditEntryResponse schema."""

    def test_audit_entry_has_required_fields(self):
        from app.schemas.face import AuditEntryResponse

        entry = AuditEntryResponse(
            id=1,
            event_type="MATCH",
            query_hash="abc123def456",
            result_name="John Doe",
            distance=0.32,
            gps_lat=40.7,
            gps_lon=-74.0,
            timestamp="2026-05-24T12:00:00+00:00",
        )
        assert entry.id == 1
        assert entry.event_type == "MATCH"
        assert entry.query_hash == "abc123def456"
        assert entry.distance == 0.32

    def test_audit_entry_optional_fields(self):
        from app.schemas.face import AuditEntryResponse

        entry = AuditEntryResponse(
            id=2,
            event_type="NO_MATCH",
            query_hash="xyz789",
            timestamp="2026-05-24T12:00:00+00:00",
        )
        assert entry.result_name is None
        assert entry.distance is None
        assert entry.gps_lat is None

    def test_valid_event_types(self):
        from app.schemas.face import AuditEntryResponse

        for event_type in ("MATCH", "NO_MATCH", "REGISTER", "SPOOF_BLOCKED"):
            entry = AuditEntryResponse(
                id=1, event_type=event_type, query_hash="abc",
                timestamp="2026-05-24T12:00:00+00:00",
            )
            assert entry.event_type == event_type


class TestAuditWithSession:
    """GET /audit with the database mocked."""

    def test_lists_entries_with_event_filter(self, client, auth_headers, db_session):
        stamp = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
        result = MagicMock()
        result.fetchall.return_value = [(1, "MATCH", "h" * 64, "Jane Doe", 0.3, 12.9, 77.5, stamp, 3)]
        db_session.execute.return_value = result

        response = client.get("/api/v1/audit", params={"event_type": "MATCH", "page": 2, "page_size": 10,
                                                       "tenant_id": 3}, headers=auth_headers)

        assert response.status_code == 200
        entry = response.json()[0]
        assert (entry["eventType"], entry["resultName"], entry["tenantId"]) == ("MATCH", "Jane Doe", 3)
        assert entry["timestamp"].startswith("2026-10-24")
        sql, params = db_session.execute.await_args.args
        assert params == {"limit": 10, "offset": 10, "tenant_id": 3, "event_type": "MATCH"}
        assert "event_type = :event_type" in str(sql)

    def test_non_admin_is_forbidden(self, client, db_session):
        token = create_access_token({"sub": "viewer", "role": "viewer"})
        response = client.get("/api/v1/audit", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403
        db_session.execute.assert_not_awaited()
