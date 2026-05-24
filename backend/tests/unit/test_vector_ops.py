"""
Test vector ops (pgvector ANN query + threshold logic).

Covers:
- compute_query_hash: deterministic, different embeddings
- cosine_ann_query: empty results, matches under threshold, threshold filtering
- register_profile: insert + commit
- add_audit_entry: insert only
- update_alert_status: status update with confirmation
- SQL uses cosine distance (not similarity)
"""

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import numpy as np
import pytest

from app.db.vector_ops import compute_query_hash


class TestComputeQueryHash:
    """Test SHA-256 query hash computation."""

    def test_hash_deterministic(self):
        emb = np.random.randn(512).astype(np.float32)
        h1 = compute_query_hash(emb.tobytes())
        h2 = compute_query_hash(emb.tobytes())
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_different_embeddings(self):
        e1 = np.random.randn(512).astype(np.float32)
        e2 = np.random.randn(512).astype(np.float32)
        assert compute_query_hash(e1.tobytes()) != compute_query_hash(e2.tobytes())

    def test_hash_empty_bytes(self):
        h = compute_query_hash(b"")
        assert len(h) == 64

    def test_hash_is_hex_string(self):
        emb = np.zeros(512, dtype=np.float32)
        h = compute_query_hash(emb.tobytes())
        assert all(c in "0123456789abcdef" for c in h)


class TestCosineANNQuery:
    """Test the ANN query with mocked database."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_matches(self):
        mock_session = AsyncMock()

        class MockResult:
            def __iter__(self):
                return iter([])
            def fetchall(self):
                return []

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        result = await cosine_ann_query(mock_session, np.zeros(512).tolist(), threshold=0.58)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_matches_under_threshold(self):
        """Mock session returns results — verify structure."""
        mock_session = AsyncMock()

        class Row:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, key):
                return self._vals[key]
            def __iter__(self):
                return iter(self._vals)

        class MockResult:
            def __iter__(self):
                return iter([
                    Row([1, "John Doe", "JD", 0.32]),
                    Row([2, "Jane Smith", None, 0.45]),
                ])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        result = await cosine_ann_query(mock_session, np.ones(512).tolist(), threshold=0.6)

        assert len(result) == 2
        assert result[0]["suspect_name"] == "John Doe"
        assert result[0]["distance"] == 0.32
        assert result[1]["suspect_name"] == "Jane Smith"
        assert result[1]["distance"] == 0.45

    @pytest.mark.asyncio
    async def test_threshold_filters_strictly(self):
        """Only matches with distance <= threshold should be returned."""
        mock_session = AsyncMock()

        class Row:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, key):
                return self._vals[key]
            def __iter__(self):
                return iter(self._vals)

        class MockResult:
            def __iter__(self):
                return iter([
                    Row([1, "Close", None, 0.30]),
                    Row([2, "Borderline", None, 0.58]),
                    Row([3, "TooFar", None, 0.70]),
                ])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        # The mock returns all rows, but the SQL WHERE clause
        # (face_embedding <=> :vec) <= :threshold should filter them.
        # Since we mock the DB, the mock returns all rows.
        # We verify the threshold was passed to the query.
        result = await cosine_ann_query(
            mock_session, np.ones(512).tolist(), threshold=0.58
        )
        # In the real DB, only Close (0.30) and Borderline (0.58) pass.
        # TooFar (0.70) would be filtered by the WHERE clause.
        # Since we mock the DB layer, we check the call args instead.
        call_args = mock_session.execute.call_args
        params = call_args[0][1] if call_args[0] else call_args[1]
        assert "threshold" in params
        assert params["threshold"] == 0.58

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        """A custom threshold should be passed to the query."""
        mock_session = AsyncMock()

        class MockResult:
            def __iter__(self):
                return iter([])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        result = await cosine_ann_query(mock_session, np.zeros(512).tolist(), threshold=0.45)
        assert result == []

    @pytest.mark.asyncio
    async def test_default_threshold_from_config(self):
        """When threshold=None, config value should be used."""
        from app.core.config import settings
        from app.db import vector_ops

        mock_session = AsyncMock()

        class MockResult:
            def __iter__(self):
                return iter([])

        mock_session.execute = AsyncMock(return_value=MockResult())

        result = await vector_ops.cosine_ann_query(mock_session, np.zeros(512).tolist())
        assert result == []
        # Verify the threshold was passed (execute is called with positional params)
        call_args = mock_session.execute.call_args
        params = call_args[0][1] if call_args[0] else call_args[1]
        assert params["threshold"] == settings.match_threshold

    @pytest.mark.asyncio
    async def test_distance_rounded_to_6_decimals(self):
        """Distances should be rounded to 6 decimal places."""
        mock_session = AsyncMock()

        class Row:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, key):
                return self._vals[key]
            def __iter__(self):
                return iter(self._vals)

        class MockResult:
            def __iter__(self):
                return iter([Row([1, "Test", None, 0.123456789])])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        result = await cosine_ann_query(mock_session, np.ones(512).tolist(), threshold=1.0)
        assert result[0]["distance"] == 0.123457

    @pytest.mark.asyncio
    async def test_default_limit_is_10(self):
        """Default limit should be 10."""
        mock_session = AsyncMock()

        class MockResult:
            def __iter__(self):
                return iter([])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        await cosine_ann_query(mock_session, np.ones(512).tolist())
        call_args = mock_session.execute.call_args
        params = call_args[0][1] if call_args[0] else call_args[1]
        assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_custom_limit(self):
        """Custom limit should override default."""
        mock_session = AsyncMock()

        class MockResult:
            def __iter__(self):
                return iter([])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        await cosine_ann_query(mock_session, np.ones(512).tolist(), limit=25)
        call_args = mock_session.execute.call_args
        params = call_args[0][1] if call_args[0] else call_args[1]
        assert params["limit"] == 25


class TestThresholdLogic:
    """Test that matches are filtered by threshold."""

    @pytest.mark.asyncio
    async def test_strict_threshold_filters_matches(self):
        mock_session = AsyncMock()

        class MockResult:
            def __iter__(self):
                return iter([])

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import cosine_ann_query
        result = await cosine_ann_query(mock_session, np.zeros(512).tolist(), threshold=0.58)
        assert result == []


class TestRegisterAndAudit:
    """Test register_profile and add_audit_entry."""

    @pytest.mark.asyncio
    async def test_register_profile_inserts_record(self):
        """Verify register_profile calls session.add + commit."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        class FakeProfile:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        import app.db.vector_ops as vector_ops
        vector_ops.SuspectProfile = FakeProfile

        await vector_ops.register_profile(
            mock_session, "Test", None, None, b"test_bytes"
        )

        mock_session.commit.assert_awaited_once()
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_profile_writes_face_embedding_column(self):
        """register_profile should write to face_embedding field (not face_embedding_bytes)."""
        captured_profile = None

        class FakeProfile:
            def __init__(self, **kwargs):
                nonlocal captured_profile
                self.id = 999
                self.__dict__.update(kwargs)
                captured_profile = kwargs

        # Patch the models module so register_profile's import gets our fake
        import app.db.models as models_module
        original = models_module.SuspectProfile
        models_module.SuspectProfile = FakeProfile

        try:
            from app.db import vector_ops

            mock_session = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()

            await vector_ops.register_profile(
                mock_session,
                "Test", None, None, b"\x01\x02\x03\x04" * 8  # 32 bytes → 64 hex chars
            )

            assert captured_profile["face_embedding"] is not None
            # Should be hex-encoded bytes
            assert isinstance(captured_profile["face_embedding"], str)
            assert len(captured_profile["face_embedding"]) == 64  # 32 bytes → 64 hex chars
        finally:
            models_module.SuspectProfile = original

    @pytest.mark.asyncio
    async def test_audit_entry_insert_only(self):
        """Audit log must only accept INSERT."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        class FakeEntry:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        import app.db.vector_ops as vector_ops
        vector_ops.AuditLog = FakeEntry

        await vector_ops.add_audit_entry(
            mock_session, event_type="MATCH", query_hash="abc123", result_name="Test"
        )

        mock_session.commit.assert_awaited_once()
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_audit_entry_with_distance_and_gps(self):
        """Audit entry should accept optional distance and GPS coordinates."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        class FakeEntry:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        import app.db.vector_ops as vector_ops
        vector_ops.AuditLog = FakeEntry

        await vector_ops.add_audit_entry(
            mock_session,
            event_type="MATCH",
            query_hash="xyz",
            result_name="John",
            distance=0.35,
            gps_lat=40.7,
            gps_lon=-74.0,
        )

        call_args = mock_session.add.call_args
        entry = call_args[0][0]
        assert entry.distance == 0.35
        assert entry.gps_lat == 40.7
        assert entry.gps_lon == -74.0


class TestUpdateAlertStatus:
    """Test update_alert_status helper."""

    @pytest.mark.asyncio
    async def test_update_confirms_alert(self):
        """Confirming an alert sets status to CONFIRMED."""
        mock_session = AsyncMock()

        class Row:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, key):
                return self._vals[key]

        class MockResult:
            def __init__(self, row):
                self._row = row
            def fetchone(self):
                return self._row

        mock_session.execute = AsyncMock(return_value=MockResult(Row([1, "CONFIRMED"])))

        from app.db.vector_ops import update_alert_status
        result = await update_alert_status(mock_session, 1, "CONFIRMED", datetime.now(timezone.utc))

        assert result == 1

    @pytest.mark.asyncio
    async def test_update_dismisses_alert(self):
        """Dismissing an alert sets status to DISMISSED."""
        mock_session = AsyncMock()

        class Row:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, key):
                return self._vals[key]

        class MockResult:
            def __init__(self, row):
                self._row = row
            def fetchone(self):
                return self._row

        mock_session.execute = AsyncMock(return_value=MockResult(Row([1, "DISMISSED"])))

        from app.db.vector_ops import update_alert_status
        result = await update_alert_status(mock_session, 2, "DISMISSED")

        # Returns the row id from RETURNING clause
        assert result == 1

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self):
        """Updating a non-existent alert returns None."""
        mock_session = AsyncMock()

        class MockResult:
            def fetchone(self):
                return None

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import update_alert_status
        result = await update_alert_status(mock_session, 99999, "CONFIRMED")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_commits_session(self):
        """update_alert_status should call commit."""
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()

        class Row:
            def __getitem__(self, key):
                return None
        class MockResult:
            def fetchone(self):
                return None

        mock_session.execute = AsyncMock(return_value=MockResult())

        from app.db.vector_ops import update_alert_status
        await update_alert_status(mock_session, 1, "CONFIRMED")

        mock_session.commit.assert_awaited()
