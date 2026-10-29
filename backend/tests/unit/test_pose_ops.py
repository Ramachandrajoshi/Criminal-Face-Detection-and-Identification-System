"""
Pose-tagged embedding storage and search (app/db/pose_ops.py), plus the
vector_ops writes not covered elsewhere, against a mocked session.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.core.config import INSIGHTFACE_EMBEDDING_VERSION
from app.core.crypto import decrypt_embedding_bytes
from app.core.pose import HeadPose
from app.db import pose_ops, vector_ops


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _returning(*ids):
    """session.execute results for consecutive INSERT … RETURNING id statements."""
    return [SimpleNamespace(scalar_one=lambda i=i: i) for i in ids]


def _rows(rows, fetchone=None):
    result = MagicMock()
    result.__iter__.return_value = iter(rows)
    result.fetchone.return_value = fetchone
    return result


LEFT = (_unit(1), HeadPose(yaw=-31.234, pitch=4.567, roll=1.0), 0.81234)
FRONT = (_unit(2), HeadPose(yaw=2.0, pitch=-1.0, roll=0.0), 0.95)


class TestRegisterProfileWithPoses:
    async def test_inserts_profile_then_each_pose_in_one_transaction(self):
        session = AsyncMock()
        session.execute.side_effect = _returning(6, 101, 102)
        template = _unit(3)

        profile_id, pose_ids = await pose_ops.register_profile_with_poses(
            session, "Jane Doe", "JD", {"gender": "F"}, template, [LEFT, FRONT], tenant_id=2,
        )

        assert (profile_id, pose_ids) == (6, [101, 102])
        profile_sql, profile_params = session.execute.await_args_list[0].args
        assert "INSERT INTO face_profiles" in str(profile_sql)
        assert profile_params["embedding_version"] == INSIGHTFACE_EMBEDDING_VERSION
        assert profile_params["tenant_id"] == 2
        np.testing.assert_allclose(decrypt_embedding_bytes(profile_params["enc"]), template, rtol=1e-6)
        assert profile_params["vec"].startswith("[") and profile_params["vec"].count(",") == 511

        pose_params = session.execute.await_args_list[1].args[1]
        assert (pose_params["face_id"], pose_params["pose_bucket"]) == (6, LEFT[1].bucket)
        assert (pose_params["yaw"], pose_params["pitch"], pose_params["quality"]) == (-31.23, 4.57, 0.8123)
        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    async def test_failure_rolls_back_and_reraises(self):
        session = AsyncMock()
        session.execute.side_effect = [*_returning(6), RuntimeError("unique violation")]
        with pytest.raises(RuntimeError):
            await pose_ops.register_profile_with_poses(session, "Jane", None, None, _unit(3), [LEFT])
        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()


class TestAddAndCountPoses:
    async def test_add_poses_to_existing_profile(self):
        session = AsyncMock()
        session.execute.side_effect = _returning(201, 202)
        assert await pose_ops.add_poses(session, 6, [LEFT, FRONT], tenant_id=3) == [201, 202]
        assert all(c.args[1]["tenant_id"] == 3 for c in session.execute.await_args_list)
        session.commit.assert_awaited_once()

    async def test_add_poses_failure_rolls_back(self):
        session = AsyncMock()
        session.execute.side_effect = RuntimeError("fk violation")
        with pytest.raises(RuntimeError):
            await pose_ops.add_poses(session, 6, [LEFT])
        session.rollback.assert_awaited_once()

    async def test_count_poses(self):
        session = AsyncMock()
        session.execute.return_value = SimpleNamespace(scalar_one=lambda: 3)
        assert await pose_ops.count_poses(session, 6, tenant_id=2) == 3
        assert session.execute.await_args.args[1] == {"face_id": 6, "tenant_id": 2}


class TestReplacePoses:
    async def test_replaces_profile_embedding_and_every_pose_in_one_transaction(self):
        session = AsyncMock()
        session.execute.side_effect = [_rows([], fetchone=(6,)), MagicMock(), *_returning(301, 302)]
        template = _unit(3)

        pose_ids = await pose_ops.replace_poses(session, 6, template, [LEFT, FRONT], tenant_id=2)

        assert pose_ids == [301, 302]
        calls = session.execute.await_args_list
        update_sql, update_params = calls[0].args
        assert "UPDATE face_profiles" in str(update_sql)
        assert (update_params["face_id"], update_params["tenant_id"]) == (6, 2)
        assert update_params["embedding_version"] == INSIGHTFACE_EMBEDDING_VERSION
        np.testing.assert_allclose(decrypt_embedding_bytes(update_params["enc"]), template, rtol=1e-6)
        delete_sql, delete_params = calls[1].args
        assert "DELETE FROM face_pose_embeddings" in str(delete_sql)
        assert delete_params == {"face_id": 6, "tenant_id": 2}
        assert [c.args[1]["pose_bucket"] for c in calls[2:]] == [LEFT[1].bucket, FRONT[1].bucket]
        assert all((c.args[1]["face_id"], c.args[1]["tenant_id"]) == (6, 2) for c in calls[2:])
        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    async def test_unknown_profile_changes_nothing(self):
        session = AsyncMock()
        session.execute.return_value = _rows([], fetchone=None)

        assert await pose_ops.replace_poses(session, 99, _unit(3), [LEFT]) is None

        assert session.execute.await_count == 1  # no DELETE or INSERT
        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()

    async def test_failure_rolls_back_so_old_poses_survive(self):
        session = AsyncMock()
        session.execute.side_effect = [_rows([], fetchone=(6,)), MagicMock(), RuntimeError("lost connection")]
        with pytest.raises(RuntimeError):
            await pose_ops.replace_poses(session, 6, _unit(3), [LEFT])
        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()


class TestListAndDeletePoses:
    async def test_list_rounds_float32_noise_and_never_returns_embeddings(self):
        created = datetime(2026, 10, 24, tzinfo=timezone.utc)
        session = AsyncMock()
        session.execute.return_value = _rows([
            (101, "left", np.float32(-31.23), np.float32(1.46), np.float32(0.5), np.float32(0.81234), created),
            (102, "frontal", None, None, None, None, None),
        ])

        poses = await pose_ops.list_poses(session, 6)

        assert poses[0] == {"id": 101, "pose_bucket": "left", "yaw": -31.23, "pitch": 1.46, "roll": 0.5,
                            "quality": 0.8123, "created_at": created.isoformat()}
        assert poses[1]["yaw"] is None and poses[1]["created_at"] == ""
        assert "embedding" not in str(session.execute.await_args.args[0]).split("FROM")[0]

    @pytest.mark.parametrize("row, deleted", [((101,), True), (None, False)])
    async def test_delete_pose(self, row, deleted):
        session = AsyncMock()
        session.execute.return_value = _rows([], fetchone=row)
        assert await pose_ops.delete_pose(session, 6, 101, tenant_id=2) is deleted
        assert session.execute.await_args.args[1] == {"pose_id": 101, "face_id": 6, "tenant_id": 2}
        session.commit.assert_awaited_once()

    async def test_delete_commit_failure_rolls_back(self):
        session = AsyncMock()
        session.execute.return_value = _rows([], fetchone=(101,))
        session.commit.side_effect = RuntimeError("serialization failure")
        with pytest.raises(RuntimeError):
            await pose_ops.delete_pose(session, 6, 101)
        session.rollback.assert_awaited_once()


class TestPoseAnnQuery:
    async def test_returns_best_pose_per_person(self):
        session = AsyncMock()
        session.execute.return_value = _rows([
            (6, "Jane Doe", "JD", np.float32(0.3123456789), "left", np.float32(-30.0), np.float32(2.0)),
            (9, "John Roe", None, 0.41, "frontal", None, None),
        ])

        matches = await pose_ops.pose_ann_query(session, _unit(4), threshold=0.6, limit=5, tenant_id=2)

        assert matches[0] == {"id": 6, "face_name": "Jane Doe", "alias": "JD", "distance": 0.312346,
                              "matched_pose": "left", "matched_yaw": -30.0, "matched_pitch": 2.0, "tenant_id": 2}
        assert matches[1]["matched_yaw"] is None and matches[1]["matched_pitch"] is None
        sql, params = session.execute.await_args.args
        assert "DISTINCT ON (p.face_id)" in str(sql)
        assert (params["threshold"], params["limit"], params["tenant_id"]) == (0.6, 5, 2)
        assert params["embedding_version"] == INSIGHTFACE_EMBEDDING_VERSION


class TestVectorOpsWrites:
    @pytest.mark.parametrize("row, updated", [((6,), True), (None, False)])
    async def test_update_face_embedding(self, row, updated):
        session = AsyncMock()
        session.execute.return_value = _rows([], fetchone=row)
        embedding = _unit(5)

        assert await vector_ops.update_face_embedding(session, 6, embedding, tenant_id=2) is updated

        params = session.execute.await_args.args[1]
        assert (params["face_id"], params["tenant_id"]) == (6, 2)
        np.testing.assert_allclose(decrypt_embedding_bytes(params["enc"]), embedding, rtol=1e-6)
        session.commit.assert_awaited_once()

    async def test_update_commit_failure_rolls_back(self):
        session = AsyncMock()
        session.execute.return_value = _rows([], fetchone=(6,))
        session.commit.side_effect = RuntimeError("lost connection")
        with pytest.raises(RuntimeError):
            await vector_ops.update_face_embedding(session, 6, _unit(5))
        session.rollback.assert_awaited_once()

    async def test_register_profile_commit_failure_rolls_back(self):
        session = AsyncMock()
        session.execute.return_value = SimpleNamespace(scalar_one=lambda: 6)
        session.commit.side_effect = RuntimeError("lost connection")
        with pytest.raises(RuntimeError):
            await vector_ops.register_profile(session, "Jane", None, None, _unit(5))
        session.rollback.assert_awaited_once()

    async def test_create_alert_is_pending_review(self):
        added = []
        session = AsyncMock()
        session.add = added.append

        async def refresh(alert):
            alert.id = 43
        session.refresh.side_effect = refresh

        alert_id = await vector_ops.create_alert(session, audit_log_id=42, face_id=6, event_type="MATCH",
                                                 distance=0.31, gps_lat=12.9, gps_lon=77.5, tenant_id=2)

        assert alert_id == 43
        (alert,) = added
        assert (alert.status, alert.audit_log_id, alert.face_id, alert.tenant_id) == ("PENDING_REVIEW", 42, 6, 2)

    @pytest.mark.parametrize("write", ["create_alert", "add_audit_entry"])
    async def test_failed_write_rolls_back(self, write):
        session = AsyncMock()
        session.add = lambda obj: None
        session.commit.side_effect = RuntimeError("lost connection")
        with pytest.raises(RuntimeError):
            if write == "create_alert":
                await vector_ops.create_alert(session, 42, 6, "MATCH", 0.3)
            else:
                await vector_ops.add_audit_entry(session, "MATCH", "h" * 64)
        session.rollback.assert_awaited_once()
