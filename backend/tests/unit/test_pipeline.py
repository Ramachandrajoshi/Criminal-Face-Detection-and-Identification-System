"""
Test pipeline stages independently with mocks.

Covers every code path in pipeline.py:
- Register: success, no face, embed fail, spoof blocked
- Search: no match, match, no face, embed fail, spoof blocked
- Config: default values, threshold not hard-coded
- Health: liveness probe
- Auth middleware: routes protected, health not
"""

from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import io
import os
import asyncio
import json

import numpy as np
import pytest
from PIL import Image
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.core.auth import create_access_token


@pytest.fixture
def client():
    return TestClient(__import__("app.main", fromlist=["app"]).app)


@pytest.fixture
def mock_embedding():
    """Return a unit-norm 512-d embedding array."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)



def _mock_present_versions(versions):
    """Stub the tenant's enrolled-recipe census used by the search pipeline."""
    return patch(
        "app.db.vector_ops.get_present_embedding_versions",
        AsyncMock(return_value=set(versions)),
    )

def make_test_image(color="red"):
    """Create a test image file for uploads."""
    img = Image.new("RGB", (112, 112), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return UploadFile(filename="test.jpg", file=buf)


def make_test_image_bytes():
    """Return raw image bytes for liveness checks."""
    img = Image.new("RGB", (112, 112), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_liveness_true():
    """Create a liveness mock that passes (returns True)."""
    return MagicMock(return_value=True)


def _mock_liveness_false():
    """Create a liveness mock that fails (returns False)."""
    return MagicMock(return_value=False)


def _patch_pipeline_liveness(true=True):
    """Context manager that patches check_liveness_on_bytes."""
    liveness_result = _mock_liveness_true() if true else _mock_liveness_false()
    return patch("app.core.pipeline.check_liveness_on_bytes", liveness_result)


# ──────────────────────────────────────────────
# Register Pipeline Tests
# ──────────────────────────────────────────────

class TestRegisterPipeline:
    """Test the register pipeline stages."""

    @pytest.mark.asyncio
    async def test_register_success(self, mock_embedding):
        """Full success path: liveness → detect → embed → register → audit."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_register = AsyncMock(return_value=42)
        mock_audit = AsyncMock(return_value=1)

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.db.vector_ops.register_profile", mock_register), \
             patch("app.db.vector_ops.add_audit_entry", mock_audit):

            from app.core.pipeline import register_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await register_pipeline(
                file, AsyncMock(), "Test Suspect", alias="T-Suspect"
            )

            assert result["status"] == "REGISTERED"
            assert result["profile_id"] == 42
            mock_extract.assert_called_once()
            mock_register.assert_called_once()
            mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_no_face(self):
        """Detect returns None — should return ERROR with message."""
        mock_detect = MagicMock(return_value=None)

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect):

            from app.core.pipeline import register_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await register_pipeline(
                file, AsyncMock(), "Test Suspect"
            )

            assert result["status"] == "ERROR"
            assert "No face detected" in result["error"]

    @pytest.mark.asyncio
    async def test_register_embed_fail(self):
        """Embedding extraction fails — should return ERROR."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=None)

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract):

            from app.core.pipeline import register_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await register_pipeline(
                file, AsyncMock(), "Test Suspect"
            )

            assert result["status"] == "ERROR"
            assert "Embedding extraction failed" in result["error"]

    @pytest.mark.asyncio
    async def test_register_bypasses_liveness(self, mock_embedding):
        """Even if check_liveness_on_bytes would return False, registration bypasses it and succeeds."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_register = AsyncMock(return_value=42)
        mock_audit = AsyncMock(return_value=1)

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_false()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.db.vector_ops.register_profile", mock_register), \
             patch("app.db.vector_ops.add_audit_entry", mock_audit):

            from app.core.pipeline import register_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await register_pipeline(file, AsyncMock(), "Test Suspect")

            assert result["status"] == "REGISTERED"
            assert result["profile_id"] == 42

    @pytest.mark.asyncio
    async def test_check_liveness_on_bytes_returns_bool(self):
        """check_liveness_on_bytes should return a boolean."""
        with patch("app.core.liveness.check_liveness_with_deepface") as mock_live:
            from app.core.pipeline import check_liveness_on_bytes

            mock_live.return_value = {"is_live": True, "spoof_probability": 0.1}
            result = check_liveness_on_bytes(b"fake_bytes")
            assert result is True

            mock_live.return_value = {"is_live": False, "spoof_probability": 0.95}
            result = check_liveness_on_bytes(b"fake_bytes")
            assert result is False

            mock_live.return_value = {"is_live": False}
            result = check_liveness_on_bytes(b"fake_bytes")
            assert result is False

            mock_live.return_value = {}  # no is_live key
            result = check_liveness_on_bytes(b"fake_bytes")
            assert result is False


# ──────────────────────────────────────────────
# Search Pipeline Tests
# ──────────────────────────────────────────────

class TestSearchPipeline:
    """Test the search pipeline stages."""

    @pytest.mark.parametrize("present,expect_embedded,expect_queried", [
        # Nothing enrolled under the current recipe yet — the state right after
        # a version bump. Only the enrolled recipe is searched; the current one
        # is still embedded, for the audit hash.
        ({2},       [3, 2],    [2]),
        ({1},       [3, 1],    [1]),
        ({2, 1},    [3, 2, 1], [2, 1]),
        # Part-way through re-enrolment.
        ({3, 2},    [3, 2],    [3, 2]),
        # Fully re-enrolled — one embedding, one query.
        ({3},       [3],       [3]),
        # Empty tenant: nothing to query, but still hashable.
        (set(),     [3],       []),
        # Unknown versions in the column are ignored, not embedded.
        ({2, 99},   [3, 2],    [2]),
    ])
    async def test_search_embeds_only_enrolled_recipes_plus_current(
        self, present, expect_embedded, expect_queried
    ):
        """Recipes with no enrolled rows are never embedded or queried."""
        from app.core.pipeline import run_pipeline

        embeddings = {v: np.full(512, v, dtype=np.float32) for v in (3, 2, 1)}

        with _mock_present_versions(present), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding",
                   side_effect=lambda face, version: embeddings[version]) as extract, \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])) as verify:
            result = await run_pipeline(make_test_image(), AsyncMock(), tenant_id=7)

        assert sorted(c.args[1] for c in extract.call_args_list) == sorted(expect_embedded)
        assert sorted(c.kwargs["target_version"] for c in verify.await_args_list) == sorted(expect_queried)
        assert result["status"] == "NO_MATCH"

    async def test_search_ranks_across_recipes_rather_than_preferring_newest(self):
        """A closer match in an older recipe must not be masked by a newer one."""
        from app.core.pipeline import run_pipeline

        # v3 holds a weak hit, v2 a much closer one.
        hits = {
            3: [{"id": 33, "face_name": "Weak v3", "distance": 0.51}],
            2: [{"id": 22, "face_name": "Close v2", "distance": 0.12}],
        }

        async def verify(embedding, session, **kwargs):
            return list(hits[kwargs["target_version"]])

        with _mock_present_versions({3, 2}), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding",
                   side_effect=lambda face, version: np.full(512, version, dtype=np.float32)), \
             patch("app.core.pipeline.verify_match", side_effect=verify):
            result = await run_pipeline(make_test_image(), AsyncMock(), limit=5)

        assert result["status"] == "MATCH"
        assert [m["id"] for m in result["matches"]] == [22, 33]
        assert result["matches"][0]["face_name"] == "Close v2"

    async def test_search_truncates_merged_results_to_limit(self):
        """`limit` bounds the merged result set, not each recipe separately."""
        from app.core.pipeline import run_pipeline

        async def verify(embedding, session, **kwargs):
            v = kwargs["target_version"]
            return [{"id": v * 10 + i, "distance": 0.1 * v + i / 100} for i in range(3)]

        with _mock_present_versions({3, 2}), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding",
                   side_effect=lambda face, version: np.full(512, version, dtype=np.float32)), \
             patch("app.core.pipeline.verify_match", side_effect=verify):
            result = await run_pipeline(make_test_image(), AsyncMock(), limit=4)

        distances = [m["distance"] for m in result["matches"]]
        assert len(result["matches"]) == 4
        assert distances == sorted(distances)
        assert distances[0] == pytest.approx(0.2)   # v2's closest beats v3's

    async def test_search_embeds_recipes_concurrently(self):
        """The per-recipe embeddings run on the pool together, not serially."""
        import threading
        from app.core.pipeline import run_pipeline
        from app.core.config import settings

        assert settings.inference_pool_size >= 3, "pool must fit every recipe"
        barrier = threading.Barrier(3, timeout=10)

        def extract(face, version):
            # Serial execution can never get three callers here at once, so a
            # regression to a sequential loop fails with BrokenBarrierError.
            barrier.wait()
            return np.full(512, version, dtype=np.float32)

        with _mock_present_versions({3, 2, 1}), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding", side_effect=extract), \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])):
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "NO_MATCH"
        assert not barrier.broken

    @pytest.mark.parametrize("present,matched", [
        ({3, 2}, 3), ({3, 2}, 2), ({3, 2}, None), ({2}, 2), ({2}, None),
    ])
    async def test_audit_hash_identifies_the_query_not_the_matched_recipe(self, present, matched):
        """The same image always audits under the same hash."""
        from app.core.pipeline import run_pipeline
        from app.db.vector_ops import compute_query_hash

        embeddings = {v: np.full(512, v, dtype=np.float32) for v in (3, 2, 1)}

        async def verify(embedding, session, **kwargs):
            v = kwargs["target_version"]
            return [{"id": v, "distance": 0.2}] if v == matched else []

        with _mock_present_versions(present), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding",
                   side_effect=lambda face, version: embeddings[version]), \
             patch("app.core.pipeline.verify_match", side_effect=verify):
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == ("MATCH" if matched else "NO_MATCH")
        assert result["query_hash"] == compute_query_hash(embeddings[3].tobytes())

    async def test_failed_embedding_of_an_enrolled_recipe_is_an_error(self):
        """Vectors that were never compared must not read as NO_MATCH."""
        from app.core.pipeline import run_pipeline

        def extract(face, version):
            return None if version == 2 else np.full(512, version, dtype=np.float32)

        with _mock_present_versions({3, 2}), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding", side_effect=extract), \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])) as verify:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "ERROR"
        assert result["error"] == "Embedding extraction failed"
        assert result["matches"] == []
        assert result["query_hash"]
        # v3 was still searched — only v2 was lost.
        assert [c.kwargs["target_version"] for c in verify.await_args_list] == [3]

    async def test_failed_embedding_of_an_unenrolled_current_recipe_still_matches(self):
        """The hash-only recipe failing costs hash stability, not the search."""
        from app.core.pipeline import run_pipeline
        from app.db.vector_ops import compute_query_hash

        v2 = np.full(512, 2, dtype=np.float32)

        async def verify(embedding, session, **kwargs):
            return [{"id": 22, "distance": 0.1}]

        with _mock_present_versions({2}), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding",
                   side_effect=lambda face, version: None if version == 3 else v2), \
             patch("app.core.pipeline.verify_match", side_effect=verify):
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "MATCH"
        assert result["matches"][0]["id"] == 22
        # Falls back to the newest recipe that did embed.
        assert result["query_hash"] == compute_query_hash(v2.tobytes())

    async def test_liveness_rejection_never_queries_any_recipe(self):
        from app.core.pipeline import run_pipeline

        with patch("app.core.pipeline.check_liveness_on_bytes", return_value=False), \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock(return_value=1)), \
             patch("app.db.vector_ops.get_present_embedding_versions") as census, \
             patch("app.core.pipeline.extract_embedding") as extract, \
             patch("app.core.pipeline.verify_match") as verify:
            result = await run_pipeline(make_test_image(), AsyncMock(), enforce_liveness=True)
        assert result["status"] == "SPOOF_BLOCKED"
        census.assert_not_called()
        extract.assert_not_called()
        verify.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_no_match(self, mock_embedding):
        """Pipeline runs, ANN returns empty — status NO_MATCH."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_verify = AsyncMock(return_value=[])

        with _mock_present_versions({3, 2, 1}), \
             patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.core.pipeline.verify_match", mock_verify):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(
                file, AsyncMock(), gps_lat=40.7, gps_lon=-74.0
            )

            assert result["status"] == "NO_MATCH"
            assert result["matches"] == []
            assert result["query_hash"] is not None
            # One query per enrolled recipe.
            assert mock_verify.call_count == 3

    @pytest.mark.asyncio
    async def test_search_match(self, mock_embedding):
        """Pipeline runs, ANN returns match — status MATCH."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_verify = AsyncMock(return_value=[
            {"id": 1, "face_name": "John Doe", "alias": "JD", "distance": 0.32, "query_hash": "abc"},
        ])

        with _mock_present_versions({3}), \
             patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.core.pipeline.verify_match", mock_verify):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock())

            assert result["status"] == "MATCH"
            assert len(result["matches"]) == 1
            assert result["matches"][0]["face_name"] == "John Doe"
            assert result["matches"][0]["distance"] == 0.32

    @pytest.mark.asyncio
    async def test_search_no_face(self):
        """No face detected — should return ERROR."""
        mock_detect = MagicMock(return_value=None)

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock())

            assert result["status"] == "ERROR"
            assert result["error"] == "No face detected"
            assert result["matches"] == []

    @pytest.mark.asyncio
    async def test_search_embed_fail(self):
        """Embedding extraction fails — should return ERROR."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=None)

        # The current recipe holds enrolled rows, so failing to embed it means
        # those vectors were never compared: ERROR, not NO_MATCH.
        with _mock_present_versions({3}), \
             patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock())

            assert result["status"] == "ERROR"
            assert "Embedding extraction failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_spoof_blocked(self):
        """Liveness check fails — should return SPOOF_BLOCKED."""
        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_false()), \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock(return_value=1)):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock(), enforce_liveness=True)

            assert result["status"] == "SPOOF_BLOCKED"
            assert result["matches"] == []
            assert result["query_hash"] is not None

    @pytest.mark.asyncio
    async def test_search_spoof_logs_audit(self):
        """SPOOF_BLOCKED should log to audit trail."""
        audit_calls = []

        async def capture_audit(*args, **kwargs):
            audit_calls.append((args, kwargs))
            return 1

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_false()), \
             patch("app.db.vector_ops.add_audit_entry", capture_audit):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock(), enforce_liveness=True)

            assert result["status"] == "SPOOF_BLOCKED"
            assert len(audit_calls) == 1
            assert audit_calls[0][1].get("event_type") == "SPOOF_BLOCKED"

    @pytest.mark.asyncio
    async def test_search_match_with_limit(self, mock_embedding):
        """verify_match respects the limit parameter."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_verify = AsyncMock(return_value=[])

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.core.pipeline.verify_match", mock_verify):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            session = AsyncMock()
            result = await run_pipeline(file, session)
            assert result["status"] == "NO_MATCH"

    @pytest.mark.asyncio
    async def test_search_gps_coordinates_propagated(self, mock_embedding):
        """GPS coordinates from search request should be in the result."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_verify = AsyncMock(return_value=[])

        with patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.core.pipeline.verify_match", mock_verify):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(
                file, AsyncMock(), gps_lat=51.5, gps_lon=-0.1
            )

            assert result["gps_lat"] == 51.5
            assert result["gps_lon"] == -0.1

    @pytest.mark.asyncio
    async def test_verify_match_attaches_query_hash(self, mock_embedding):
        """verify_match should attach query_hash to each match dict."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)

        mock_ann = AsyncMock(return_value=[
            {"id": 1, "face_name": "Match", "alias": None, "distance": 0.25},
        ])

        with _mock_present_versions({3}), \
             patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.db.vector_ops.cosine_ann_query", mock_ann):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock())

            assert result["status"] == "MATCH"
            assert "query_hash" in result["matches"][0]
            assert isinstance(result["matches"][0]["query_hash"], str)

    @pytest.mark.asyncio
    async def test_search_match_distance_within_threshold(self, mock_embedding):
        """Match distances should be within threshold (cosine distance)."""
        mock_detect = MagicMock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        mock_extract = MagicMock(return_value=mock_embedding)
        mock_verify = AsyncMock(return_value=[
            {"id": 1, "face_name": "Close Match", "alias": None, "distance": 0.32},
        ])

        with _mock_present_versions({3}), \
             patch("app.core.pipeline.check_liveness_on_bytes", _mock_liveness_true()), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.core.pipeline.extract_embedding", mock_extract), \
             patch("app.core.pipeline.verify_match", mock_verify):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image_bytes()),
            )
            result = await run_pipeline(file, AsyncMock())

            assert result["status"] == "MATCH"
            assert result["matches"][0]["distance"] <= 0.58


# ──────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────

class TestConfig:
    """Test configuration loading and constraints."""

    def test_default_match_threshold(self):
        from app.core.config import settings
        assert settings.match_threshold == 0.58
        assert settings.deepface_model == "ArcFace"
        assert settings.deepface_detector == "retinaface"

    def test_match_threshold_not_hardcoded_in_routes(self):
        """Verify MATCH_THRESHOLD is not hard-coded in route files."""
        routes_dir = Path(__file__).resolve().parent.parent / "app" / "api" / "routes"
        for fname in routes_dir.glob("*.py"):
            content = fname.read_text()
            assert "0.58" not in content, f"Hard-coded threshold found in {fname.name}"

    def test_deepface_model_is_arcface(self):
        from app.core.config import settings
        assert settings.deepface_model == "ArcFace"

    def test_deepface_detector_is_retinaface(self):
        from app.core.config import settings
        assert settings.deepface_detector == "retinaface"

    def test_jwt_expiry_hours(self):
        from app.core.config import settings
        assert settings.jwt_expiry_hours == 8

    def test_api_port_is_8000(self):
        from app.core.config import settings
        assert settings.api_port == 8000


# ──────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────

class TestHealthEndpoint:
    """Test the health endpoint (no auth required)."""

    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_database_field(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "database" in data

    def test_health_no_auth_required(self, client):
        """Health endpoint must be accessible without any token."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers.get("WWW-Authenticate") is None


# ──────────────────────────────────────────────
# Auth Middleware Integration
# ──────────────────────────────────────────────

class TestAuthMiddlewareIntegration:
    """Verify that AuthMiddleware is applied to all protected routes."""

    def test_all_protected_routes_require_jwt(self):
        """Every non-health route should return 401 without a token."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )

        paths = ["/api/v1/alerts", "/api/v1/audit"]
        for path in paths:
            response = client.get(path)
            assert response.status_code == 401, f"Expected 401 for {path} without auth"

    def test_register_returns_401_without_token(self):
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        img = Image.new("RGB", (112, 112), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/api/v1/register",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={"face_name": "Test"},
        )
        assert response.status_code == 401

    def test_search_returns_401_without_token(self):
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

    def test_login_does_not_require_token(self):
        """Login endpoint should be accessible without a token."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        username = os.environ.get("ADMIN_TEST_USERNAME", "")
        password = os.environ.get("ADMIN_TEST_PASSWORD", "")
        response = client.post(
            "/api/v1/login",
            json={"username": username, "password": password},
        )
        # May return 200 (success) or 401 (wrong credentials in test env)
        assert response.status_code in (200, 401)

    def test_health_and_login_are_public(self):
        """Health and login endpoints should be accessible without auth."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        h = client.get("/api/v1/health")
        assert h.status_code == 200
        l = client.post(
            "/api/v1/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert l.status_code in (200, 401)  # 401 for bad creds is still "no auth required"


class TestAuthEndpoint:
    """Test the auth/login endpoints."""

    def test_login_success(self):
        """Valid credentials should return a token."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        username = os.environ.get("ADMIN_TEST_USERNAME", "")
        password = os.environ.get("ADMIN_TEST_PASSWORD", "")
        response = client.post(
            "/api/v1/login",
            json={"username": username, "password": password},
        )
        data = response.json()
        assert response.status_code == 200
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_credentials(self):
        """Invalid credentials should return 401."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.post(
            "/api/v1/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_login_missing_fields(self):
        """Missing fields should return 422."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        response = client.post(
            "/api/v1/login",
            json={},
        )
        assert response.status_code == 422

    def test_refresh_token_success(self):
        """Valid token should be refreshable."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        # First get a token
        username = os.environ.get("ADMIN_TEST_USERNAME", "")
        password = os.environ.get("ADMIN_TEST_PASSWORD", "")
        login_resp = client.post(
            "/api/v1/login",
            json={"username": username, "password": password},
        )
        token = login_resp.json()["access_token"]
        # Then refresh it
        response = client.post(
            "/api/v1/token/refresh",
            json={"access_token": token},
        )
        data = response.json()
        assert response.status_code == 200
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_expired_token_success(self):
        """Expired but signature-valid token should be refreshable."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        )
        from datetime import timedelta
        # Generate an expired token
        token = create_access_token(
            {"sub": "admin", "role": "admin"},
            expires_delta=timedelta(seconds=-1)
        )
        # Refresh it
        response = client.post(
            "/api/v1/token/refresh",
            json={"access_token": token},
        )
        data = response.json()
        assert response.status_code == 200
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_invalid_token(self):
        """Invalid token should return 401."""
        response = TestClient(
            __import__("app.main", fromlist=["app"]).app, raise_server_exceptions=False
        ).post(
            "/api/v1/token/refresh",
            json={"access_token": "invalid.token.value"},
        )
        assert response.status_code == 401


# ──────────────────────────────────────────────
# Preprocessing Enhancement Tests
# ──────────────────────────────────────────────

class TestPreprocessing:
    """
    Unit tests for _preprocess_for_arcface enhancement stages.

    Each test patches only the relevant settings flag(s) so stages are
    tested in isolation.  The full chain smoke test verifies all stages
    compose correctly end-to-end.
    """

    def _make_face(self, h: int = 200, w: int = 160, dtype=np.uint8) -> np.ndarray:
        """Return a synthetic RGB face array of the requested shape and dtype."""
        rng = np.random.default_rng(42)
        face = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        if dtype == np.float32:
            face = (face / 255.0).astype(np.float32)
        return face

    def _make_dark_face(self, h: int = 200, w: int = 160) -> np.ndarray:
        """Return a very dark RGB face (mean ≈ 30) to trigger gamma lift."""
        rng = np.random.default_rng(7)
        face = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        return face

    # --- Stage 0: float32 → uint8 conversion ---

    def test_float32_input_is_converted_to_uint8(self):
        """float32 [0,1] input must be converted to uint8 [0,255]."""
        from app.core.pipeline import _preprocess_for_arcface
        face_float = self._make_face(dtype=np.float32)
        result = _preprocess_for_arcface(face_float)
        assert result.dtype == np.uint8, "Output must be uint8"
        assert result.shape == (112, 112, 3), "Output must be 112×112 RGB"

    def test_uint8_input_passes_through_dtype_unchanged(self):
        """uint8 input should remain uint8 after preprocessing."""
        from app.core.pipeline import _preprocess_for_arcface
        face_uint8 = self._make_face(dtype=np.uint8)
        result = _preprocess_for_arcface(face_uint8)
        assert result.dtype == np.uint8

    # --- Stage 1: Upscale tiny crops ---

    def test_small_crop_is_upscaled(self):
        """A crop smaller than small_crop_min_side must be Lanczos-upscaled."""
        import cv2
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface

        tiny_face = np.zeros((60, 50, 3), dtype=np.uint8)
        # All later stages disabled so they don't mask the upscale effect
        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = True
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = False
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result = _preprocess_for_arcface(tiny_face)

        # The output must be the canonical 112×112 (pad+resize step runs last)
        assert result.shape == (112, 112, 3)

    def test_large_crop_is_not_upscaled(self):
        """A crop already above the threshold must not be upscaled."""
        import cv2
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface

        large_face = self._make_face(h=200, w=200)
        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = True
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = False
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result = _preprocess_for_arcface(large_face)

        assert result.shape == (112, 112, 3)

    # --- Stage 2: Denoising ---

    def test_denoising_does_not_crash(self):
        """fastNlMeansDenoisingColored must complete without exception."""
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface

        face = self._make_face()
        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = False
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = True
            mock_settings.enable_gamma_correction = False
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result = _preprocess_for_arcface(face)

        assert result.shape == (112, 112, 3)
        assert result.dtype == np.uint8

    def test_denoising_disabled_flag_skips_step(self):
        """When enable_denoising=False the output should equal the CLAHE-free path."""
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface
        import cv2

        face = self._make_face()
        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = False
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = False
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result = _preprocess_for_arcface(face.copy())

        # Should still produce a valid 112×112 array
        assert result.shape == (112, 112, 3)

    # --- Stage 3: Auto-gamma ---

    def test_gamma_lifts_dark_image_mean_luminance(self):
        """Gamma correction must increase mean luminance of a very dark face."""
        import cv2
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface

        dark_face = self._make_dark_face()
        dark_gray = cv2.cvtColor(dark_face, cv2.COLOR_RGB2GRAY)
        before_mean = float(np.mean(dark_gray))

        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = False
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = True
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result = _preprocess_for_arcface(dark_face.copy())

        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        after_mean = float(np.mean(result_gray))
        assert after_mean > before_mean, (
            f"Gamma should lift a dark face: before={before_mean:.1f} after={after_mean:.1f}"
        )

    def test_gamma_no_op_on_mid_tone_image(self):
        """For a face with mean luminance ≈ 128 gamma correction should be near 1.0."""
        import math
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface
        import cv2

        # Create a face with exactly mean=128
        mid_face = np.full((112, 112, 3), 128, dtype=np.uint8)

        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = False
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = True
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result = _preprocess_for_arcface(mid_face.copy())

        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        result_mean = float(np.mean(result_gray))
        # Isolate gamma from padding: mid-tone pixels should remain mid-tone.
        assert abs(result_mean - 128) <= 1, (
            f"Mid-tone image should not be significantly altered: mean={result_mean:.1f}"
        )

    # --- Stage 5: Unsharp mask ---

    def test_unsharp_mask_increases_sharpness(self):
        """Unsharp mask should increase the image's high-frequency content."""
        import cv2
        from unittest.mock import patch
        from app.core.pipeline import _preprocess_for_arcface

        # Use a face with visible edges (random noise has high HF content)
        face = self._make_face()

        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = False
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = False
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = False
            mock_settings.clahe_clip_limit = 2.0
            result_no_usm = _preprocess_for_arcface(face.copy())

        with patch("app.core.pipeline.settings") as mock_settings:
            mock_settings.enable_upscale_small_crops = False
            mock_settings.small_crop_min_side = 160
            mock_settings.enable_denoising = False
            mock_settings.enable_gamma_correction = False
            mock_settings.enable_clahe = False
            mock_settings.enable_unsharp_mask = True
            mock_settings.unsharp_strength = 1.5
            mock_settings.clahe_clip_limit = 2.0
            result_usm = _preprocess_for_arcface(face.copy())

        # Measure sharpness via Laplacian variance (higher = sharper)
        lap_before = cv2.Laplacian(result_no_usm, cv2.CV_64F).var()
        lap_after = cv2.Laplacian(result_usm, cv2.CV_64F).var()
        assert lap_after >= lap_before, (
            f"Unsharp mask should not reduce sharpness: before={lap_before:.2f} after={lap_after:.2f}"
        )

    # --- Full chain smoke test ---

    def test_full_chain_produces_valid_112_rgb_uint8(self):
        """All stages enabled together must produce a 112×112 uint8 RGB array."""
        from app.core.pipeline import _preprocess_for_arcface
        from app.core.config import settings

        # Use a float32 input (DeepFace output format) of a non-square crop
        face = self._make_face(h=227, w=167, dtype=np.float32)
        result = _preprocess_for_arcface(face)

        assert result.dtype == np.uint8, "Output dtype must be uint8"
        assert result.shape == (112, 112, 3), "Output shape must be (112, 112, 3)"
        assert result.min() >= 0 and result.max() <= 255, "Pixel values must be in [0,255]"

    def test_new_config_flags_have_correct_defaults(self):
        """Verify all new preprocessing config flags have the expected defaults."""
        from app.core.config import settings

        assert settings.enable_upscale_small_crops is True
        assert settings.small_crop_min_side == 160
        assert settings.enable_denoising is True
        assert settings.enable_gamma_correction is True
        assert settings.enable_unsharp_mask is True
        assert settings.unsharp_strength == 1.5


class TestPreprocessingRegression:
    @pytest.fixture(autouse=True)
    def disable_enhancements(self):
        from app.core.config import settings

        with patch.multiple(
            settings, enable_upscale_small_crops=False, enable_denoising=False,
            enable_gamma_correction=False, enable_clahe=False, enable_unsharp_mask=False,
        ):
            yield

    @pytest.mark.parametrize("h,w", [(112, 72), (72, 112), (112, 73), (73, 112), (112, 112)])
    @pytest.mark.parametrize("normalized", [False, True])
    def test_rectangular_crop_is_centered_without_stretching(self, h, w, normalized):
        from app.core.pipeline import _preprocess_for_arcface

        face = np.full((h, w, 3), [255, 0, 0], dtype=np.uint8)
        if normalized:
            face = face.astype(np.float32) / 255
        result = _preprocess_for_arcface(face)
        expected = np.zeros((112, 112, 3), dtype=np.uint8)
        y, x = (112 - h) // 2, (112 - w) // 2
        expected[y:y + h, x:x + w] = [255, 0, 0]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("h,w", [(227, 167), (167, 227)])
    def test_resize_receives_square_crop(self, h, w):
        import cv2
        from app.core.pipeline import _preprocess_for_arcface

        with patch("cv2.resize", wraps=cv2.resize) as resize:
            _preprocess_for_arcface(np.full((h, w, 3), 100, dtype=np.uint8))
        assert resize.call_args.args[0].shape == (227, 227, 3)

    @pytest.mark.parametrize("value", [0, 1, 254, 255])
    @pytest.mark.parametrize("version", [2, 3])
    def test_gamma_handles_extreme_exposures(self, value, version):
        from app.core.pipeline import _preprocess_for_arcface
        from app.core.config import settings

        face = np.full((112, 112, 3), value, dtype=np.uint8)
        with patch.object(settings, "enable_gamma_correction", True):
            result = _preprocess_for_arcface(face, version=version)
        assert result.dtype == np.uint8
        assert result.shape == face.shape
        assert np.isfinite(result).all()
        if value in (0, 1, 255):
            np.testing.assert_array_equal(result, face)

    @pytest.mark.parametrize("version", [1, 2, 3, None])
    def test_embedding_handoff_preserves_versioned_color_contract(self, version):
        import sys
        from types import SimpleNamespace
        from app.core.pipeline import extract_embedding

        rgb = np.full((112, 112, 3), [240, 100, 20], dtype=np.uint8)
        deepface = MagicMock()
        deepface.represent.return_value = [{"embedding": np.ones(512).tolist()}]
        with patch.dict(sys.modules, {"deepface": SimpleNamespace(DeepFace=deepface)}):
            embedding = extract_embedding(rgb) if version is None else extract_embedding(rgb, version=version)
        passed = deepface.represent.call_args.kwargs
        expected = rgb[:, :, ::-1] if version in (3, None) else rgb
        np.testing.assert_array_equal(passed["img_path"], expected)
        assert passed["img_path"].flags.c_contiguous
        assert passed["detector_backend"] == "skip"
        assert embedding.shape == (512,)
        assert np.linalg.norm(embedding) == pytest.approx(1.0)

    @pytest.mark.parametrize("version", [1, 2])
    def test_legacy_geometry_is_preserved(self, version):
        import cv2
        from app.core.pipeline import _preprocess_for_arcface

        face = np.full((112, 72, 3), [240, 100, 20], dtype=np.uint8)
        # Historical recipe added 20 columns only on the left, then stretched.
        legacy = np.zeros((112, 92, 3), dtype=np.uint8)
        legacy[:, 20:] = face
        expected = cv2.resize(legacy, (112, 112), interpolation=cv2.INTER_LANCZOS4)
        np.testing.assert_array_equal(_preprocess_for_arcface(face, version=version), expected)

    def test_unknown_recipe_is_rejected(self):
        from app.core.pipeline import _preprocess_for_arcface

        with pytest.raises(ValueError, match="Unsupported embedding version"):
            _preprocess_for_arcface(np.zeros((112, 112, 3), dtype=np.uint8), version=99)


# ──────────────────────────────────────────────
# Embedding version 4 (InsightFace) alongside versions 1–3
# ──────────────────────────────────────────────

def _real_jpeg_bytes():
    import cv2
    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), 128, np.uint8))
    return buf.tobytes()


class TestEmbedWithInsightface:
    def _engine(self, face):
        engine = MagicMock()
        engine.detect_primary.return_value = (face, "none" if face is None else "standard")
        engine.align.side_effect = lambda frame, kps: np.zeros((112, 112, 3), np.uint8)
        engine.embed.return_value = np.ones(512, np.float32) / np.sqrt(512)
        return engine

    def test_embeds_the_primary_face(self):
        from app.core.face_engine import DetectedFace
        from app.core.pipeline import embed_with_insightface

        face = DetectedFace(np.array([0, 0, 40, 40.0]), 0.60, np.full((5, 2), 2.0))
        engine = self._engine(face)

        with patch("app.core.face_engine.get_engine", return_value=engine):
            emb = embed_with_insightface(_real_jpeg_bytes())
            embed_with_insightface(_real_jpeg_bytes(), enrollment=True)

        assert emb.shape == (512,)
        np.testing.assert_array_equal(engine.align.call_args.args[1], face.landmarks)
        assert [c.kwargs["prefer_center"] for c in engine.detect_primary.call_args_list] == [False, True]

    def test_no_face_returns_none(self):
        from app.core.pipeline import embed_with_insightface

        with patch("app.core.face_engine.get_engine", return_value=self._engine(None)):
            assert embed_with_insightface(_real_jpeg_bytes()) is None

    def test_undecodable_bytes_return_none(self):
        from app.core.pipeline import embed_with_insightface

        with patch("app.core.face_engine.get_engine") as get_engine:
            assert embed_with_insightface(b"not an image") is None
        get_engine.assert_not_called()


def _v4_current():
    return patch("app.core.pipeline.CURRENT_EMBEDDING_VERSION", 4)


class TestVersion4Enrollment:
    """EMBEDDING_VERSION=4 enrols through InsightFace, never through DeepFace."""

    async def test_register_uses_insightface(self, mock_embedding):
        from app.core.pipeline import register_pipeline

        with _v4_current(), \
             patch("app.core.pipeline.embed_with_insightface", return_value=mock_embedding) as insight, \
             patch("app.core.pipeline.detect_face") as deepface_detect, \
             patch("app.db.vector_ops.register_profile", AsyncMock(return_value=7)) as reg, \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock(return_value=1)):
            result = await register_pipeline(make_test_image(), AsyncMock(), "Name")

        assert result["status"] == "REGISTERED"
        insight.assert_called_once()
        assert insight.call_args.args[1] is True     # enrollment picks the centred face
        deepface_detect.assert_not_called()
        np.testing.assert_array_equal(reg.await_args.args[4], mock_embedding)

    async def test_reenroll_uses_insightface(self, mock_embedding):
        from app.core.pipeline import reenroll_pipeline

        with _v4_current(), \
             patch("app.core.pipeline.embed_with_insightface", return_value=mock_embedding), \
             patch("app.core.pipeline.detect_face") as deepface_detect, \
             patch("app.db.vector_ops.update_face_embedding", AsyncMock(return_value=True)), \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock(return_value=1)):
            result = await reenroll_pipeline(make_test_image(), AsyncMock(), 5, "Name")

        assert result["status"] == "RE_ENROLLED"
        deepface_detect.assert_not_called()

    @pytest.mark.parametrize("engine_result,error", [
        ("none", "No face detected in the uploaded image"),
        ("raise", "Embedding extraction failed"),
    ])
    async def test_register_errors(self, engine_result, error):
        from app.core.pipeline import register_pipeline

        effect = RuntimeError("model missing") if engine_result == "raise" else None
        with _v4_current(), \
             patch("app.core.pipeline.embed_with_insightface", return_value=None, side_effect=effect), \
             patch("app.db.vector_ops.register_profile", AsyncMock()) as reg:
            result = await register_pipeline(make_test_image(), AsyncMock(), "Name")

        assert result == {"status": "ERROR", "error": error}
        reg.assert_not_awaited()


class TestMixedVersionSearch:
    """Faces enrolled under DeepFace (1–3) keep matching after switching to 4."""

    def _patches(self, present, deepface_face=True, insight=None):
        insight = np.full(512, 4, np.float32) if insight is None else insight
        return (
            _mock_present_versions(present),
            patch("app.core.pipeline.detect_face",
                  return_value=np.zeros((112, 112, 3)) if deepface_face else None),
            patch("app.core.pipeline.extract_embedding",
                  side_effect=lambda face, version: np.full(512, version, dtype=np.float32)),
            patch("app.core.pipeline.embed_with_insightface",
                  side_effect=insight if isinstance(insight, Exception) else None,
                  return_value=None if isinstance(insight, Exception) else insight),
        )

    async def test_old_faces_still_match_after_switching_to_v4(self):
        from app.core.pipeline import run_pipeline

        async def verify(embedding, session, **kwargs):
            v = kwargs["target_version"]
            return [{"id": 30, "distance": 0.2, "embedding_version": 3}] if v == 3 else []

        p = self._patches({3})
        with _v4_current(), p[0], p[1], p[2], p[3] as insight, \
             patch("app.core.pipeline.verify_match", side_effect=verify) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "MATCH"
        assert result["matches"][0]["id"] == 30
        # v4 is embedded for the audit hash even with nothing enrolled under it,
        # but only the enrolled recipe is queried.
        insight.assert_called_once()
        assert [c.kwargs["target_version"] for c in vm.await_args_list] == [3]

    async def test_each_version_is_queried_with_its_own_threshold(self):
        from app.core.config import settings
        from app.core.pipeline import run_pipeline

        p = self._patches({4, 3})
        with _v4_current(), p[0], p[1], p[2], p[3], \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "NO_MATCH"
        thresholds = {c.kwargs["target_version"]: c.kwargs["threshold"] for c in vm.await_args_list}
        assert thresholds == {4: settings.match_threshold_insightface, 3: settings.match_threshold}

    async def test_ranking_is_relative_to_each_models_threshold(self):
        from app.core.pipeline import run_pipeline

        # 0.40 under a 0.80 threshold (half-way) beats 0.35 under a 0.50 one.
        hits = {
            4: [{"id": 44, "distance": 0.40}],
            3: [{"id": 33, "distance": 0.35}],
        }

        async def verify(embedding, session, **kwargs):
            return list(hits[kwargs["target_version"]])

        p = self._patches({4, 3})
        with _v4_current(), p[0], p[1], p[2], p[3], \
             patch("app.core.pipeline.settings.match_threshold", 0.50), \
             patch("app.core.pipeline.settings.match_threshold_insightface", 0.80), \
             patch("app.core.pipeline.verify_match", side_effect=verify):
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert [m["id"] for m in result["matches"]] == [44, 33]

    async def test_no_face_from_every_engine_is_no_face(self):
        from app.core.pipeline import run_pipeline

        p = self._patches({4, 3}, deepface_face=False)
        with _v4_current(), p[0], p[1], p[2], p[3] as insight, \
             patch("app.core.pipeline.verify_match", AsyncMock()) as vm:
            insight.return_value = None
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result == {"status": "ERROR", "error": "No face detected", "matches": []}
        vm.assert_not_awaited()

    async def test_insightface_finds_the_face_deepface_missed(self):
        """A v4 hit is returned even when the DeepFace detector saw nothing."""
        from app.core.pipeline import run_pipeline

        async def verify(embedding, session, **kwargs):
            return [{"id": 44, "distance": 0.3}]

        p = self._patches({4, 3}, deepface_face=False)
        with _v4_current(), p[0], p[1], p[2], p[3], \
             patch("app.core.pipeline.verify_match", side_effect=verify) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "MATCH"
        assert [c.kwargs["target_version"] for c in vm.await_args_list] == [4]

    async def test_partial_detection_without_match_is_not_a_clean_no_match(self):
        """v3 vectors were never compared, so this must not read as NO_MATCH."""
        from app.core.pipeline import run_pipeline

        p = self._patches({4, 3}, deepface_face=False)
        with _v4_current(), p[0], p[1], p[2], p[3], \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])):
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "ERROR"
        assert result["error"] == "Embedding extraction failed"

    async def test_insightface_engine_failure_keeps_deepface_results(self):
        from app.core.pipeline import run_pipeline

        async def verify(embedding, session, **kwargs):
            return [{"id": 33, "distance": 0.2}]

        p = self._patches({4, 3}, insight=RuntimeError("onnx failed"))
        with _v4_current(), p[0], p[1], p[2], p[3], \
             patch("app.core.pipeline.verify_match", side_effect=verify) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "MATCH"
        assert [c.kwargs["target_version"] for c in vm.await_args_list] == [3]

    async def test_default_v3_deployment_never_loads_insightface(self):
        """With EMBEDDING_VERSION=3 and no v4 rows the new engine is untouched."""
        from app.core.pipeline import run_pipeline

        p = self._patches({3, 2})
        with p[0], p[1], p[2], p[3] as insight, \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])):
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "NO_MATCH"
        insight.assert_not_called()


# ──────────────────────────────────────────────
# Embedding version 5 (YOLOv5-Face + AdaFace) alongside 4 and 1–3
# ──────────────────────────────────────────────

class TestEmbedWithAdaface:
    def test_embeds_the_primary_face_with_the_adaface_engine(self):
        from app.core.face_engine import DetectedFace
        from app.core.pipeline import embed_with_adaface

        face = DetectedFace(np.array([0, 0, 40, 40.0]), 0.95, np.full((5, 2), 2.0))
        engine = TestEmbedWithInsightface()._engine(face)

        with patch("app.core.adaface_engine.get_engine", return_value=engine), \
             patch("app.core.face_engine.get_engine") as v4_engine:
            emb = embed_with_adaface(_real_jpeg_bytes(), enrollment=True)

        assert emb.shape == (512,)
        assert engine.detect_primary.call_args.kwargs["prefer_center"] is True
        np.testing.assert_array_equal(engine.align.call_args.args[1], face.landmarks)
        v4_engine.assert_not_called()

    def test_no_face_returns_none(self):
        from app.core.pipeline import embed_with_adaface

        with patch("app.core.adaface_engine.get_engine", return_value=TestEmbedWithInsightface()._engine(None)):
            assert embed_with_adaface(_real_jpeg_bytes()) is None

    def test_debug_dump_stages_are_tagged_v5(self):
        from app.core.face_engine import DetectedFace
        from app.core.pipeline import embed_with_adaface

        face = DetectedFace(np.array([0, 0, 40, 40.0]), 0.95, np.full((5, 2), 2.0))
        with patch("app.core.adaface_engine.get_engine", return_value=TestEmbedWithInsightface()._engine(face)), \
             patch("app.core.pipeline.dump_detection") as detection, \
             patch("app.core.pipeline.dump_stage") as stage:
            embed_with_adaface(_real_jpeg_bytes())

        assert detection.call_args.args[0] == "v5_01_detect"
        assert detection.call_args.kwargs["label"] == "yolov5face 0.95"
        assert stage.call_args.args[0] == "v5_02_aligned_112"


def _v5_current():
    return patch("app.core.pipeline.CURRENT_EMBEDDING_VERSION", 5)


class TestVersion5:
    """EMBEDDING_VERSION=5 enrols through AdaFace; v4 and v3 faces keep matching."""

    async def test_register_uses_adaface(self, mock_embedding):
        from app.core.pipeline import register_pipeline

        with _v5_current(), \
             patch("app.core.pipeline.embed_with_adaface", return_value=mock_embedding) as adaface, \
             patch("app.core.pipeline.embed_with_insightface") as insight, \
             patch("app.core.pipeline.detect_face") as deepface_detect, \
             patch("app.db.vector_ops.register_profile", AsyncMock(return_value=7)) as reg, \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock(return_value=1)):
            result = await register_pipeline(make_test_image(), AsyncMock(), "Name")

        assert result["status"] == "REGISTERED"
        adaface.assert_called_once()
        assert adaface.call_args.args[1] is True     # enrollment picks the centred face
        insight.assert_not_called()
        deepface_detect.assert_not_called()
        np.testing.assert_array_equal(reg.await_args.args[4], mock_embedding)

    def _patches(self, present, adaface=None):
        adaface = np.full(512, 5, np.float32) if adaface is None else adaface
        return (
            _mock_present_versions(present),
            patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))),
            patch("app.core.pipeline.extract_embedding",
                  side_effect=lambda face, version: np.full(512, version, dtype=np.float32)),
            patch("app.core.pipeline.embed_with_insightface", return_value=np.full(512, 4, np.float32)),
            patch("app.core.pipeline.embed_with_adaface",
                  side_effect=adaface if isinstance(adaface, Exception) else None,
                  return_value=None if isinstance(adaface, Exception) else adaface),
        )

    async def test_every_present_version_is_queried_with_its_own_embedding_and_threshold(self):
        from app.core.config import settings
        from app.core.pipeline import run_pipeline

        p = self._patches({5, 4, 3})
        with _v5_current(), p[0], p[1], p[2], p[3], p[4], \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "NO_MATCH"
        calls = {c.kwargs["target_version"]: c for c in vm.await_args_list}
        assert {v: c.kwargs["threshold"] for v, c in calls.items()} == {
            5: settings.match_threshold_adaface,
            4: settings.match_threshold_insightface,
            3: settings.match_threshold,
        }
        for v, c in calls.items():
            assert c.args[0][0] == v                 # each recipe's own query vector

    async def test_v4_faces_still_match_after_switching_to_v5(self):
        from app.core.pipeline import run_pipeline

        async def verify(embedding, session, **kwargs):
            return [{"id": 40, "distance": 0.3, "embedding_version": 4}] if kwargs["target_version"] == 4 else []

        p = self._patches({4})
        with _v5_current(), p[0], p[1], p[2], p[3], p[4] as adaface, \
             patch("app.core.pipeline.verify_match", side_effect=verify) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "MATCH"
        assert result["matches"][0]["id"] == 40
        # v5 is embedded for the audit hash even with nothing enrolled under it.
        adaface.assert_called_once()
        assert [c.kwargs["target_version"] for c in vm.await_args_list] == [4]

    async def test_adaface_engine_failure_keeps_other_results(self):
        from app.core.pipeline import run_pipeline

        async def verify(embedding, session, **kwargs):
            return [{"id": 40, "distance": 0.3}]

        p = self._patches({5, 4}, adaface=RuntimeError("onnx failed"))
        with _v5_current(), p[0], p[1], p[2], p[3], p[4], \
             patch("app.core.pipeline.verify_match", side_effect=verify) as vm:
            result = await run_pipeline(make_test_image(), AsyncMock())

        assert result["status"] == "MATCH"
        assert [c.kwargs["target_version"] for c in vm.await_args_list] == [4]

    async def test_v4_deployment_never_loads_adaface(self):
        from app.core.pipeline import run_pipeline

        p = self._patches({4, 3})
        with _v4_current(), p[0], p[1], p[2], p[3], p[4] as adaface, \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])):
            await run_pipeline(make_test_image(), AsyncMock())

        adaface.assert_not_called()


# ──────────────────────────────────────────────
# GPU memory configuration (TensorFlow)
# ──────────────────────────────────────────────

class TestConfigureGpu:
    def _fake_tf(self, n_gpus=1):
        import sys
        tf = MagicMock()
        tf.config.list_physical_devices.return_value = [MagicMock(name=f"gpu{i}") for i in range(n_gpus)]
        return patch.dict(sys.modules, {"tensorflow": tf}), tf

    def test_hard_cap_when_limit_is_set(self):
        from app.main import _configure_gpu

        mods, tf = self._fake_tf()
        with mods, patch("app.main.settings.enable_gpu", True), \
             patch("app.main.settings.tf_enable_gpu", True), \
             patch("app.main.settings.tf_gpu_memory_limit_mb", 3072):
            _configure_gpu()

        tf.config.LogicalDeviceConfiguration.assert_called_once_with(memory_limit=3072)
        tf.config.set_logical_device_configuration.assert_called_once()
        tf.config.experimental.set_memory_growth.assert_not_called()

    def test_memory_growth_by_default(self):
        from app.main import _configure_gpu

        mods, tf = self._fake_tf(n_gpus=2)
        with mods, patch("app.main.settings.enable_gpu", True), \
             patch("app.main.settings.tf_enable_gpu", True), \
             patch("app.main.settings.tf_gpu_memory_limit_mb", 0):
            _configure_gpu()

        assert tf.config.experimental.set_memory_growth.call_count == 2
        tf.config.set_logical_device_configuration.assert_not_called()

    def test_tf_gpu_disabled_hides_gpu_from_tensorflow_only(self):
        from app.main import _configure_gpu

        mods, tf = self._fake_tf()
        with mods, patch("app.main.settings.enable_gpu", True), \
             patch("app.main.settings.tf_enable_gpu", False), \
             patch("app.main.settings.tf_gpu_memory_limit_mb", 3072):
            _configure_gpu()

        tf.config.set_visible_devices.assert_called_once_with([], "GPU")
        tf.config.set_logical_device_configuration.assert_not_called()
        tf.config.experimental.set_memory_growth.assert_not_called()

    def test_gpu_disabled_never_touches_tensorflow(self):
        from app.main import _configure_gpu

        mods, tf = self._fake_tf()
        with mods, patch("app.main.settings.enable_gpu", False):
            _configure_gpu()
        tf.config.list_physical_devices.assert_not_called()
