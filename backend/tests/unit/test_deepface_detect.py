"""
DeepFace detection and embedding (recipes 1–3) in app/core/pipeline.py, the
re-enrolment error paths, and the debug face dump wiring of the pipeline.
"""

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

from app.core import debug_dump, pipeline
from app.core.debug_dump import face_dump


def _jpeg() -> bytes:
    return cv2.imencode(".jpg", np.full((120, 120, 3), 90, np.uint8))[1].tobytes()


def _upload() -> UploadFile:
    buf = io.BytesIO()
    Image.new("RGB", (112, 112), color="red").save(buf, format="JPEG")
    buf.seek(0)
    return UploadFile(filename="probe.jpg", file=buf)


def _face(w, h, confidence=0.99, x=10, y=10):
    return {
        "face": np.full((h, w, 3), 0.5, np.float32),
        "facial_area": {"x": x, "y": y, "w": w, "h": h, "left_eye": (x + w // 3, y + h // 3),
                        "right_eye": (x + 2 * w // 3, y + h // 3)},
        "confidence": confidence,
    }


def _deepface(by_backend):
    """DeepFace stub whose extract_faces answers per detector backend."""
    def extract_faces(img_path, detector_backend, **kwargs):
        outcome = by_backend[detector_backend]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    deepface = MagicMock()
    deepface.extract_faces.side_effect = extract_faces
    return deepface, patch.dict(sys.modules, {"deepface": SimpleNamespace(DeepFace=deepface)})


def _backends(primary, mtcnn, opencv):
    return {pipeline.settings.deepface_detector: primary, "mtcnn": mtcnn, "opencv": opencv}


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_dump.settings, "debug_face_dump", True)
    monkeypatch.setattr(debug_dump.settings, "debug_face_dump_dir", str(tmp_path))
    return tmp_path


def _manifest(dump):
    return json.loads((dump.dir / "manifest.json").read_text(encoding="utf-8"))


class TestDetectFace:
    def test_undecodable_bytes_return_none_without_loading_deepface(self):
        deepface, stub = _deepface({})
        with stub:
            assert pipeline.detect_face(b"not an image") is None
        deepface.extract_faces.assert_not_called()

    def test_largest_confident_face_from_first_backend(self):
        small, large = _face(30, 30), _face(60, 50)
        deepface, stub = _deepface(_backends([small, large], AssertionError, AssertionError))
        with stub:
            face = pipeline.detect_face(_jpeg())
        assert face is large["face"]
        assert deepface.extract_faces.call_count == 1
        assert deepface.extract_faces.call_args.kwargs["enforce_detection"] is True

    def test_low_confidence_and_errors_fall_through_to_the_next_backend(self):
        weak, good = _face(60, 60, confidence=0.4), _face(40, 40)
        _, stub = _deepface(_backends([weak], RuntimeError("mtcnn weights missing"), [good]))
        with stub:
            assert pipeline.detect_face(_jpeg()) is good["face"]

    def test_no_face_anywhere_returns_none(self):
        no_face = ValueError("Face could not be detected")
        deepface, stub = _deepface(_backends(no_face, no_face, no_face))
        with stub:
            assert pipeline.detect_face(_jpeg()) is None
        assert deepface.extract_faces.call_count == 3

    def test_debug_dump_records_attempts_and_the_chosen_box(self, dump_dir):
        _, stub = _deepface(_backends(ValueError("none"), [_face(60, 60, confidence=0.5)], [_face(40, 40)]))
        with stub, face_dump("search") as dump:
            pipeline.detect_face(_jpeg())

        manifest = _manifest(dump)
        assert [a["backend"] for a in manifest["deepface_detect_attempts"]] == [
            pipeline.settings.deepface_detector, "mtcnn", "opencv"]
        assert manifest["deepface_detect_attempts"][1]["rejected"] == "below 0.7"
        assert manifest["deepface_detect_attempts"][2]["chosen"] is True
        assert {s["file"] for s in manifest["stages"]} == {"dfc_01_detect.png", "dfc_02_crop.png"}

    def test_debug_dump_notes_total_failure(self, dump_dir):
        _, stub = _deepface(_backends(ValueError("none"), ValueError("none"), RuntimeError("cv2 broke")))
        with stub, face_dump("search") as dump:
            assert pipeline.detect_face(_jpeg()) is None
        assert _manifest(dump)["deepface_detect_attempts"][2] == {"backend": "opencv", "error": "cv2 broke"}


class TestExtractEmbeddingFailures:
    @pytest.mark.parametrize("represent", [
        MagicMock(side_effect=RuntimeError("model not loaded")),
        MagicMock(return_value=[]),
    ])
    def test_failed_representation_returns_none(self, represent):
        deepface = SimpleNamespace(represent=represent)
        with patch.dict(sys.modules, {"deepface": SimpleNamespace(DeepFace=deepface)}):
            assert pipeline.extract_embedding(np.full((112, 112, 3), 120, np.uint8), version=3) is None


class TestReenrollErrors:
    async def test_no_face_leaves_the_stored_embedding(self):
        with patch("app.core.pipeline._embed_for_enrollment",
                   AsyncMock(return_value=(None, "No face detected in the uploaded image"))), \
             patch("app.db.vector_ops.update_face_embedding", AsyncMock()) as update:
            result = await pipeline.reenroll_pipeline(_upload(), AsyncMock(), 6, "Jane Doe")
        assert result == {"status": "ERROR", "error": "No face detected in the uploaded image"}
        update.assert_not_awaited()

    async def test_unknown_face_is_an_error_and_not_audited(self):
        with patch("app.core.pipeline._embed_for_enrollment",
                   AsyncMock(return_value=(np.ones(512, np.float32), None))), \
             patch("app.db.vector_ops.update_face_embedding", AsyncMock(return_value=False)), \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock()) as audit:
            result = await pipeline.reenroll_pipeline(_upload(), AsyncMock(), 6, "Jane Doe", tenant_id=2)
        assert result == {"status": "ERROR", "error": "Face 6 not found in database"}
        audit.assert_not_awaited()


class TestSearchDebugDump:
    async def test_search_dump_records_result_and_nearest_ignoring_threshold(self, dump_dir):
        embedding = np.full(512, 1 / np.sqrt(512), np.float32)
        nearest = [{"id": 6, "distance": 0.71, "embedding_version": 3, "face_name": "Jane Doe"}]

        with patch("app.db.vector_ops.get_present_embedding_versions", AsyncMock(return_value={3})), \
             patch("app.core.pipeline.CURRENT_EMBEDDING_VERSION", 3), \
             patch("app.core.pipeline.detect_face", return_value=np.zeros((112, 112, 3))), \
             patch("app.core.pipeline.extract_embedding", return_value=embedding), \
             patch("app.core.pipeline.verify_match", AsyncMock(return_value=[])), \
             patch("app.db.vector_ops.cosine_ann_query", AsyncMock(return_value=nearest)) as ann:
            result = await pipeline.run_pipeline(_upload(), AsyncMock(), tenant_id=2)

        assert result["status"] == "NO_MATCH"
        (run_dir,) = dump_dir.iterdir()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["result"] == {"status": "NO_MATCH", "error": None, "matches": []}
        assert manifest["nearest_ignoring_threshold"]["v3"]["nearest"] == [
            {"face_id": 6, "embedding_version": 3, "distance": 0.71}]           # no names in the dump
        assert ann.await_args.kwargs["threshold"] == 2.0 and ann.await_args.kwargs["tenant_id"] == 2
        assert (run_dir / "00_input.jpg").exists()

    async def test_nearest_skips_recipes_that_failed(self, dump_dir):
        with face_dump("search") as dump, \
             patch("app.db.vector_ops.cosine_ann_query", AsyncMock(return_value=[])) as ann:
            await pipeline._dump_nearest(AsyncMock(), {4: None, 3: np.ones(512, np.float32)}, tenant_id=1)
        assert list(_manifest(dump)["nearest_ignoring_threshold"]) == ["v3"]
        assert ann.await_count == 1
