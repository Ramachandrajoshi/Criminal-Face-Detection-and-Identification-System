"""
Face profile routes (app/api/routes/register.py) with the pipeline and the
database mocked: single and batch registration, list / get / update / delete,
re-enrolment and the test-image lookup.
"""

import io
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import register as register_routes
from app.core.auth import create_access_token
from app.main import app

REGISTERED = {"status": "REGISTERED", "profile_id": 5, "query_hash": "h" * 64, "embedding_dim": 512, "tenant_id": 1}


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {create_access_token({'sub': 'operator', 'role': 'admin'})}"}


def _jpeg(size: int = 112, color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _sse(text: str) -> list[dict]:
    return [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: ")]


def _per_file_sessions(session):
    """Stand-in for async_session_factory: every batch file gets ``session``."""
    @asynccontextmanager
    async def factory():
        yield session
    return factory


def _result(**methods):
    result = MagicMock()
    for name, value in methods.items():
        getattr(result, name).return_value = value
    return result


def _profile_row(id=6, face_name="Jane Doe", created_at=datetime(2026, 10, 1, tzinfo=timezone.utc)):
    return SimpleNamespace(id=id, face_name=face_name, alias="JD", demographics={"age_band": "25-34"},
                           embedding_version=4, created_at=created_at, tenant_id=1)


class TestNameFromFilename:
    @pytest.mark.parametrize("filename, expected", [
        ("john_doe.jpg", "John Doe"),
        ("Jane-Smith-02.png", "Jane Smith 02"),
        ("face123.jpeg", "Face123"),
        ("___.jpg", "Unknown"),
    ])
    def test_name_from_filename(self, filename, expected):
        assert register_routes._name_from_filename(filename) == expected


class TestRegisterSingle:
    def test_registers_and_derives_name_from_filename(self, client, auth_headers, db_session):
        pipeline = AsyncMock(return_value=REGISTERED)
        with patch("app.api.routes.register.register_pipeline", pipeline):
            response = client.post(
                "/api/v1/register",
                files={"file": ("jane_doe.jpg", _jpeg(), "image/jpeg")},
                data={"alias": "JD", "demographics": json.dumps({"gender": "F"}), "tenant_id": "3"},
                headers=auth_headers,
            )

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["status"] == "REGISTERED" and body["profileId"] == 5 and body["embeddingDim"] == 512
        args, kwargs = pipeline.await_args
        assert args[2:5] == ("Jane Doe", "JD", {"gender": "F"})
        assert kwargs["tenant_id"] == 3

    def test_explicit_name_wins_over_filename(self, client, auth_headers, db_session):
        pipeline = AsyncMock(return_value=REGISTERED)
        with patch("app.api.routes.register.register_pipeline", pipeline):
            client.post("/api/v1/register", files={"file": ("x.jpg", _jpeg(), "image/jpeg")},
                        data={"face_name": "  John Roe "}, headers=auth_headers)
        assert pipeline.await_args.args[2] == "John Roe"

    @pytest.mark.parametrize("upload, detail", [
        (("notes.txt", b"hello", "text/plain"), "Unsupported MIME type"),
        (("big.jpg", _jpeg() + b"\0" * (5 * 1024 * 1024), "image/jpeg"), "exceeds 5 MB"),
        (("tiny.jpg", _jpeg(size=32), "image/jpeg"), "at least 64x64"),
        (("broken.jpg", b"\xff\xd8 not really a jpeg", "image/jpeg"), "Invalid image data"),
    ])
    def test_invalid_upload_is_rejected(self, client, auth_headers, db_session, upload, detail):
        response = client.post("/api/v1/register", files={"file": upload}, headers=auth_headers)
        assert response.status_code == 400
        assert detail in response.json()["detail"]

    def test_invalid_demographics_json_is_rejected(self, client, auth_headers, db_session):
        response = client.post("/api/v1/register", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                               data={"demographics": "{not json"}, headers=auth_headers)
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid demographics JSON"

    @pytest.mark.parametrize("status", ["ERROR", "SPOOF_BLOCKED"])
    def test_pipeline_failure_returns_422(self, client, auth_headers, db_session, status):
        pipeline = AsyncMock(return_value={"status": status, "error": "No face detected in the uploaded image"})
        with patch("app.api.routes.register.register_pipeline", pipeline):
            response = client.post("/api/v1/register", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                                   headers=auth_headers)
        assert response.status_code == 422
        assert response.json()["detail"] == "No face detected in the uploaded image"


class TestRegisterBatch:
    def _post(self, client, auth_headers, files, pipeline, path="/api/v1/register/batch", **data):
        session = AsyncMock()
        with patch("app.api.routes.register.register_pipeline", pipeline), \
             patch("app.api.routes.register.async_session_factory", _per_file_sessions(session)):
            return client.post(path, files=[("files", f) for f in files], data=data, headers=auth_headers), session

    def test_reports_each_file_in_input_order(self, client, auth_headers):
        async def pipeline(file, session, face_name, *args, **kwargs):
            if face_name == "No Face":
                return {"status": "ERROR", "error": "No face detected in the uploaded image"}
            return {**REGISTERED, "profile_id": 7}

        files = [("jane_doe.jpg", _jpeg(), "image/jpeg"), ("notes.txt", b"x", "text/plain"),
                 ("no_face.jpg", _jpeg(color="blue"), "image/jpeg")]
        response, _ = self._post(client, auth_headers, files, pipeline, demographics=json.dumps({"gender": "M"}))

        assert response.status_code == 201, response.text
        body = response.json()
        assert (body["totalFiles"], body["registered"], body["failed"]) == (3, 1, 2)
        assert [r["filename"] for r in body["results"]] == ["jane_doe.jpg", "notes.txt", "no_face.jpg"]
        assert body["results"][0] == {"filename": "jane_doe.jpg", "status": "REGISTERED", "profileId": 7,
                                      "faceName": "Jane Doe", "error": None}
        assert "Unsupported MIME type" in body["results"][1]["error"]
        assert body["results"][2]["status"] == "ERROR"

    def test_pipeline_exception_becomes_error_result_and_rolls_back(self, client, auth_headers):
        response, session = self._post(client, auth_headers, [("a.jpg", _jpeg(), "image/jpeg")],
                                       AsyncMock(side_effect=RuntimeError("db down")))
        result = response.json()["results"][0]
        assert (result["status"], result["error"]) == ("ERROR", "db down")
        session.rollback.assert_awaited_once()

    @pytest.mark.parametrize("path", ["/api/v1/register/batch", "/api/v1/register/batch/stream"])
    def test_rejects_more_than_50_files(self, client, auth_headers, path):
        files = [(f"f{i}.jpg", b"x", "image/jpeg") for i in range(51)]
        response, _ = self._post(client, auth_headers, files, AsyncMock(), path=path)
        assert response.status_code == 400
        assert "50 images" in response.json()["detail"]

    @pytest.mark.parametrize("path", ["/api/v1/register/batch", "/api/v1/register/batch/stream"])
    def test_rejects_invalid_demographics_json(self, client, auth_headers, path):
        response, _ = self._post(client, auth_headers, [("a.jpg", _jpeg(), "image/jpeg")], AsyncMock(),
                                 path=path, demographics="[oops")
        assert response.status_code == 400

    def test_stream_emits_start_progress_and_done(self, client, auth_headers):
        async def pipeline(file, session, face_name, *args, **kwargs):
            if face_name == "Spoof":
                return {"status": "SPOOF_BLOCKED", "error": "spoof"}
            return REGISTERED

        files = [("jane.jpg", _jpeg(), "image/jpeg"), ("spoof.jpg", _jpeg(color="blue"), "image/jpeg")]
        response, _ = self._post(client, auth_headers, files, pipeline, path="/api/v1/register/batch/stream")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse(response.text)
        assert events[0] == {"type": "start", "total": 2, "processed": 0, "elapsedMs": 0}
        progress = {e["filename"]: e for e in events if e["type"] == "progress"}
        assert progress["jane.jpg"]["status"] == "REGISTERED" and progress["jane.jpg"]["profileId"] == 5
        assert progress["spoof.jpg"]["status"] == "SPOOF_BLOCKED" and progress["spoof.jpg"]["error"] == "spoof"
        done = events[-1]
        assert (done["type"], done["processed"], done["registered"], done["failed"]) == ("done", 2, 1, 1)


class TestFaceCrud:
    def test_list_faces(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(all=[_profile_row(), _profile_row(id=2, created_at=None)])

        response = client.get("/api/v1/faces", params={"tenant_id": 2}, headers=auth_headers)

        assert response.status_code == 200
        faces = response.json()
        assert [f["id"] for f in faces] == [6, 2]
        assert faces[0]["faceName"] == "Jane Doe" and faces[0]["createdAt"].startswith("2026-10-01")
        assert faces[1]["createdAt"] == ""
        assert "face_embedding" not in json.dumps(faces)

    def test_get_face(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(one_or_none=_profile_row())
        response = client.get("/api/v1/faces/6", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["alias"] == "JD"

    def test_get_missing_face_is_404(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(one_or_none=None)
        assert client.get("/api/v1/faces/404", headers=auth_headers).status_code == 404

    def test_json_update_sets_demographics(self, client, auth_headers, db_session):
        face = SimpleNamespace(id=6, face_name="Jane", alias="J", demographics=None,
                               embedding_version=4, created_at=None, tenant_id=1)
        db_session.execute.return_value = _result(scalar_one_or_none=face)

        response = client.patch("/api/v1/faces/6", json={"alias": "   ", "demographics": {"gender": "F"}},
                                headers=auth_headers)

        assert response.status_code == 200, response.text
        assert face.alias is None                        # blank alias clears it
        assert face.demographics == {"gender": "F"}
        assert face.face_name == "Jane"                  # untouched when not sent

    def test_json_update_of_missing_face_is_404(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(scalar_one_or_none=None)
        response = client.patch("/api/v1/faces/9", json={"alias": "x"}, headers=auth_headers)
        assert response.status_code == 404
        db_session.commit.assert_not_awaited()

    def test_delete_writes_audit_entry_before_deleting(self, client, auth_headers, db_session):
        calls = []
        audit = AsyncMock(side_effect=lambda *a, **k: calls.append("audit"))
        db_session.execute.side_effect = lambda *a, **k: calls.append("execute") or _result(one_or_none=_profile_row())

        with patch("app.api.routes.register.add_audit_entry", audit):
            response = client.delete("/api/v1/faces/6", headers=auth_headers)

        assert response.status_code == 204
        assert calls == ["execute", "audit", "execute"]  # look up, audit, then delete
        assert audit.await_args.kwargs["event_type"] == "REGISTER_DELETE"
        assert audit.await_args.kwargs["result_name"] is None   # no PII in the audit log
        db_session.commit.assert_awaited_once()

    @pytest.mark.parametrize("form, tenant", [(None, 1), ({"tenant_id": "2"}, 2)])
    def test_delete_is_audited_under_the_faces_tenant(self, client, auth_headers, db_session, form, tenant):
        db_session.execute.return_value = _result(one_or_none=_profile_row())

        with patch("app.api.routes.register.add_audit_entry", AsyncMock()) as audit:
            response = client.request("DELETE", "/api/v1/faces/6", data=form, headers=auth_headers)

        assert response.status_code == 204
        assert audit.await_args.kwargs["tenant_id"] == tenant
        delete = db_session.execute.await_args_list[-1].args[0]
        assert delete.compile().params == {"id_1": 6, "tenant_id_1": tenant}

    def test_delete_missing_face_is_404(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(one_or_none=None)
        with patch("app.api.routes.register.add_audit_entry", AsyncMock()) as audit:
            response = client.delete("/api/v1/faces/6", headers=auth_headers)
        assert response.status_code == 404
        audit.assert_not_awaited()


class TestReenroll:
    def _put(self, client, auth_headers, upload=("new.jpg", _jpeg(), "image/jpeg")):
        return client.put("/api/v1/faces/6/face", files={"file": upload}, headers=auth_headers)

    def test_reenrolls_existing_face(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(one_or_none=_profile_row())
        pipeline = AsyncMock(return_value={"status": "RE_ENROLLED", "profile_id": 6,
                                           "query_hash": "q" * 64, "embedding_dim": 512})
        with patch("app.api.routes.register.reenroll_pipeline", pipeline):
            response = self._put(client, auth_headers)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "RE_ENROLLED" and body["profileId"] == 6 and body["updatedAt"]
        assert pipeline.await_args.args[2:4] == (6, "Jane Doe")

    def test_invalid_image_is_rejected(self, client, auth_headers, db_session):
        response = self._put(client, auth_headers, upload=("a.gif", b"GIF89a", "image/gif"))
        assert response.status_code == 400

    def test_missing_face_is_404(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(one_or_none=None)
        with patch("app.api.routes.register.reenroll_pipeline", AsyncMock()) as pipeline:
            response = self._put(client, auth_headers)
        assert response.status_code == 404
        pipeline.assert_not_awaited()

    def test_pipeline_error_is_422(self, client, auth_headers, db_session):
        db_session.execute.return_value = _result(one_or_none=_profile_row())
        pipeline = AsyncMock(return_value={"status": "ERROR", "error": "No face detected in the uploaded image"})
        with patch("app.api.routes.register.reenroll_pipeline", pipeline):
            response = self._put(client, auth_headers)
        assert response.status_code == 422
        assert response.json()["detail"] == "No face detected in the uploaded image"


class TestFaceTestImage:
    @pytest.fixture
    def testdata(self, tmp_path, monkeypatch):
        # The route looks for <backend root>/testdata next to app/.
        monkeypatch.setattr(register_routes, "__file__", str(tmp_path / "app" / "api" / "routes" / "register.py"))
        return tmp_path / "testdata"

    def test_missing_testdata_directory_is_404(self, client, auth_headers, testdata):
        response = client.get("/api/v1/faces/image/Jane Doe", headers=auth_headers)
        assert response.status_code == 404
        assert response.json()["detail"] == "Testdata directory not found"

    @pytest.mark.parametrize("stored, requested", [
        ("Jane_Doe.jpg", "jane doe"),       # spaces ↔ underscores, any case
        ("jane-doe.png", "jane-doe"),        # exact stem with a dash
        ("jane-doe.png", "jane_doe"),        # dash ↔ underscore fallback
    ])
    def test_finds_image_by_name(self, client, auth_headers, testdata, stored, requested):
        testdata.mkdir()
        (testdata / stored).write_bytes(_jpeg())
        response = client.get(f"/api/v1/faces/image/{requested}", headers=auth_headers)
        assert response.status_code == 200
        assert response.content == _jpeg()

    def test_unknown_name_is_404(self, client, auth_headers, testdata):
        testdata.mkdir()
        (testdata / "someone_else.jpg").write_bytes(_jpeg())
        response = client.get("/api/v1/faces/image/jane", headers=auth_headers)
        assert response.status_code == 404
        assert response.json()["detail"] == "Face image not found in testdata"
