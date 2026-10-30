"""
YOLOv5-Face + AdaFace engine (embedding version 5) — ONNX sessions are mocked.

The real-model check at the bottom runs only when ADAFACE_TEST_MODEL_DIR
points at a directory holding yolov5m_face.onnx and adaface_ir_101.onnx.
"""

import hashlib
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.core import adaface_engine
from app.core.adaface_engine import (
    DETECTOR_FILE,
    MODEL_FILES,
    RECOGNIZER_FILE,
    AdaFaceEngine,
    ensure_model_files,
)
from app.core.face_engine import InsightFaceEngine


# ──────────────────────────────────────────────
# Model files
# ──────────────────────────────────────────────

def _write_models(directory: Path) -> None:
    for name in (DETECTOR_FILE, RECOGNIZER_FILE):
        (directory / name).write_bytes(b"onnx")


def _pinned(payloads: dict):
    """MODEL_FILES with each file's checksum replaced by that of ``payloads[name]``."""
    return {name: (url, hashlib.sha256(payloads[name]).hexdigest()) for name, (url, _) in MODEL_FILES.items()}


def test_ensure_model_files_is_a_no_op_when_present(tmp_path):
    _write_models(tmp_path)
    with patch("urllib.request.urlopen") as urlopen:
        ensure_model_files(tmp_path)
    urlopen.assert_not_called()


def test_ensure_model_files_refuses_when_download_disabled(tmp_path):
    with patch.object(adaface_engine.settings, "adaface_auto_download", False):
        with pytest.raises(FileNotFoundError, match="ADAFACE_AUTO_DOWNLOAD"):
            ensure_model_files(tmp_path)


def test_ensure_model_files_downloads_only_the_missing_files(tmp_path):
    payloads = {name: f"weights:{name}".encode() for name in MODEL_FILES}
    (tmp_path / DETECTOR_FILE).write_bytes(b"already here")
    urls = {url: name for name, (url, _) in MODEL_FILES.items()}

    with patch.object(adaface_engine, "MODEL_FILES", _pinned(payloads)), \
         patch("urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(payloads[urls[url]])) as urlopen:
        ensure_model_files(tmp_path)

    assert [urls[c.args[0]] for c in urlopen.call_args_list] == [RECOGNIZER_FILE]
    assert (tmp_path / RECOGNIZER_FILE).read_bytes() == payloads[RECOGNIZER_FILE]
    assert (tmp_path / DETECTOR_FILE).read_bytes() == b"already here"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(MODEL_FILES)   # no temp files left


def test_ensure_model_files_rejects_checksum_mismatch(tmp_path):
    with patch("urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(b"tampered")):
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            ensure_model_files(tmp_path)
    assert not (tmp_path / DETECTOR_FILE).exists()
    assert not (tmp_path / RECOGNIZER_FILE).exists()


# ──────────────────────────────────────────────
# Engine with mocked ONNX sessions
# ──────────────────────────────────────────────

def _session(outputs):
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock(name="input")]
    sess.get_providers.return_value = ["CPUExecutionProvider"]
    sess.run.side_effect = lambda names, feed: outputs(feed)
    return sess


def _yolo_output(rows):
    """Fake YOLOv5-Face output: rows of (cx, cy, w, h, score, 5 landmark (x, y)) in canvas pixels."""
    out = np.zeros((1, max(len(rows), 1), 16), np.float32)
    for i, (box, score, kps) in enumerate(rows):
        out[0, i, :4], out[0, i, 4], out[0, i, 5:15], out[0, i, 15] = box, score, np.ravel(kps), 1.0
    return [out]


@pytest.fixture
def engine_factory(tmp_path):
    _write_models(tmp_path)

    def make(det_outputs, rec_output=None):
        det = _session(lambda feed: det_outputs)
        rec = _session(lambda feed: [rec_output if rec_output is not None else np.ones((1, 512), np.float32)])
        sessions = {DETECTOR_FILE: det, RECOGNIZER_FILE: rec}
        with patch("onnxruntime.InferenceSession", side_effect=lambda path, providers: sessions[Path(path).name]),              patch.object(adaface_engine.settings, "enable_gpu", False):
            return AdaFaceEngine(tmp_path, det_threshold=0.5), det, rec

    return make


def test_detect_decodes_boxes_and_landmarks_into_source_pixels(engine_factory):
    outputs = _yolo_output([
        ((320, 320, 100, 50), 0.9, np.tile([300, 310], 5)),
        ((100, 100, 100, 50), 0.3, np.zeros(10)),          # below threshold
    ])
    engine, det, _ = engine_factory(outputs)
    img = np.zeros((64, 128, 3), np.uint8)
    img[..., 0] = 255                                       # blue in BGR
    # 64×128 fills the 640 input at ×5 → 320×640, centred: 160 px grey above and below.
    faces = engine.detect(img)

    assert len(faces) == 1
    face = faces[0]
    assert face.score == pytest.approx(0.9)
    np.testing.assert_allclose(face.bbox, [54, 27, 74, 37], atol=1e-4)
    np.testing.assert_allclose(face.landmarks, np.tile([60, 30], (5, 1)), atol=1e-4)

    # RGB / 255 on a grey (114) letterbox.
    blob = det.run.call_args.args[1][engine._det_input]
    assert blob.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(blob[0, :, 0, 0], 114 / 255, atol=1e-6)
    np.testing.assert_allclose(blob[0, :, 320, 320], [0, 0, 1], atol=1e-6)


def test_native_size_keeps_a_small_image_unscaled(engine_factory):
    engine, det, _ = engine_factory(_yolo_output([((320, 320, 20, 20), 0.9, np.full(10, 320.0))]))
    faces = engine.detect(np.zeros((40, 40, 3), np.uint8), upscale=False)
    # The 40×40 image sits unscaled at the centre: canvas (300, 300) is its origin.
    np.testing.assert_allclose(faces[0].bbox, [10, 10, 30, 30], atol=1e-4)
    np.testing.assert_allclose(faces[0].landmarks, np.full((5, 2), 20.0), atol=1e-4)


def test_input_is_fixed_and_larger_passes_run_as_overlapping_tiles(engine_factory):
    engine, det, _ = engine_factory(_yolo_output([((320, 320, 40, 40), 0.9, np.full(10, 320.0))]))
    assert engine.det_size == 640

    # The "large" pass: 640×640 fitted into 1280 → ×2, tiles at 0 / 480 / 640 each way.
    faces = engine.detect(np.zeros((640, 640, 3), np.uint8), size=1280)

    assert det.run.call_count == 9
    assert all(c.args[1][engine._det_input].shape == (1, 3, 640, 640) for c in det.run.call_args_list)
    boxes = sorted(f.bbox.tolist() for f in faces)
    # Tile (0, 0) → big-image box 300–340 → ÷2; tile (640, 640) → 940–980 → ÷2.
    assert boxes[0] == pytest.approx([150, 150, 170, 170])
    assert boxes[-1] == pytest.approx([470, 470, 490, 490])
    assert len(faces) == 9


def test_detect_returns_nothing_without_faces(engine_factory):
    engine, _, _ = engine_factory(_yolo_output([]))
    assert engine.detect(np.zeros((480, 640, 3), np.uint8)) == []
    assert engine.detect(np.zeros((1080, 1920, 3), np.uint8), size=1280) == []


def test_embed_keeps_bgr_and_normalises(engine_factory):
    raw = np.arange(512, dtype=np.float32).reshape(1, 512)
    engine, _, rec = engine_factory(_yolo_output([]), rec_output=raw)

    face = np.zeros((112, 112, 3), np.uint8)
    face[..., 0] = 255                           # blue in BGR
    emb = engine.embed(face)

    assert emb.shape == (512,)
    assert np.linalg.norm(emb) == pytest.approx(1.0)
    blob = rec.run.call_args.args[1][engine._rec_input]
    # AdaFace was trained on BGR: blue stays in channel 0; (255 − 127.5) / 127.5 == 1.
    assert blob[0, 0].max() == pytest.approx(1.0)
    assert blob[0, 2].max() == pytest.approx(-1.0)


def test_face_choice_and_fallbacks_are_version_4s(engine_factory):
    """detect_primary, its passes and align come from InsightFaceEngine unchanged."""
    engine, _, _ = engine_factory(_yolo_output([]))
    for method in ("detect_primary", "_detect_passes", "align"):
        assert getattr(AdaFaceEngine, method) is getattr(InsightFaceEngine, method)

    # No face on the standard pass → the native-size pass for a small image.
    with patch.object(engine, "detect", return_value=[]) as detect:
        assert engine.detect_primary(np.zeros((40, 40, 3), np.uint8)) == (None, "none")
    assert detect.call_args_list[1].kwargs == {"upscale": False}


def test_get_engine_is_a_singleton():
    sentinel = MagicMock()
    with patch.object(adaface_engine, "_engine", None), \
         patch.object(adaface_engine, "AdaFaceEngine", return_value=sentinel) as cls:
        assert adaface_engine.get_engine() is sentinel
        assert adaface_engine.get_engine() is sentinel
    cls.assert_called_once()


# ──────────────────────────────────────────────
# Real weights (opt-in)
# ──────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("ADAFACE_TEST_MODEL_DIR"),
    reason="set ADAFACE_TEST_MODEL_DIR to run against real YOLOv5-Face + AdaFace weights",
)
def test_real_models_load_and_embed_blank_frame():
    with patch.object(adaface_engine.settings, "enable_gpu", False):
        engine = AdaFaceEngine(Path(os.environ["ADAFACE_TEST_MODEL_DIR"]))
    frame = np.zeros((480, 640, 3), np.uint8)
    assert engine.detect(frame) == []
    assert engine.detect(frame, size=1280) == []
    emb = engine.embed(cv2.resize(frame, (112, 112)))
    assert emb.shape == (512,)
