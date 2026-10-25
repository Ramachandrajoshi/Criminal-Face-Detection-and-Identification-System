"""
InsightFace engine (embedding version 4) — ONNX sessions are mocked.

The real-model check at the bottom runs only when INSIGHTFACE_TEST_MODEL_DIR
points at a directory holding det_10g.onnx and w600k_r50.onnx.
"""

import hashlib
import io
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.core import face_engine
from app.core.face_engine import (
    ARCFACE_TEMPLATE,
    DETECTOR_FILE,
    RECOGNIZER_FILE,
    InsightFaceEngine,
    _nms,
    _similarity_transform,
    ensure_model_files,
)


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────

def test_similarity_transform_recovers_known_transform():
    angle, scale, shift = np.deg2rad(17.0), 1.7, np.array([12.0, -5.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = ARCFACE_TEMPLATE.astype(np.float64)
    dst = (scale * rot @ src.T).T + shift

    m = _similarity_transform(src, dst)

    np.testing.assert_allclose(m[:, :2], scale * rot, atol=1e-9)
    np.testing.assert_allclose(m[:, 2], shift, atol=1e-9)


def test_similarity_transform_never_mirrors():
    """A reflected point set is fitted with a rotation, not a flip."""
    src = ARCFACE_TEMPLATE.astype(np.float64)
    dst = src * np.array([-1.0, 1.0])
    m = _similarity_transform(src, dst)
    assert np.linalg.det(m[:, :2]) > 0


def test_nms_suppresses_overlaps_and_keeps_best_first():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.7], dtype=np.float32)
    assert _nms(boxes, scores, 0.4) == [1, 2]


# ──────────────────────────────────────────────
# Model files
# ──────────────────────────────────────────────

def _write_models(directory: Path) -> None:
    for name in (DETECTOR_FILE, RECOGNIZER_FILE):
        (directory / name).write_bytes(b"onnx")


def _pack_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in (DETECTOR_FILE, RECOGNIZER_FILE, "genderage.onnx"):
            zf.writestr(name, f"weights:{name}")
    return buf.getvalue()


def test_ensure_model_files_is_a_no_op_when_present(tmp_path):
    _write_models(tmp_path)
    with patch("urllib.request.urlopen") as urlopen:
        ensure_model_files(tmp_path)
    urlopen.assert_not_called()


def test_ensure_model_files_refuses_when_download_disabled(tmp_path):
    with patch.object(face_engine.settings, "insightface_auto_download", False):
        with pytest.raises(FileNotFoundError, match="INSIGHTFACE_AUTO_DOWNLOAD"):
            ensure_model_files(tmp_path)


def test_ensure_model_files_downloads_and_extracts_only_needed_files(tmp_path):
    pack = _pack_bytes()
    with patch.object(face_engine, "MODEL_PACK_SHA256", hashlib.sha256(pack).hexdigest()), \
         patch("urllib.request.urlopen", return_value=io.BytesIO(pack)):
        ensure_model_files(tmp_path / "models")

    files = sorted(p.name for p in (tmp_path / "models").iterdir())
    assert files == sorted([DETECTOR_FILE, RECOGNIZER_FILE])
    assert (tmp_path / "models" / DETECTOR_FILE).read_text() == f"weights:{DETECTOR_FILE}"


def test_ensure_model_files_rejects_checksum_mismatch(tmp_path):
    with patch("urllib.request.urlopen", return_value=io.BytesIO(_pack_bytes())):
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            ensure_model_files(tmp_path)
    assert not (tmp_path / DETECTOR_FILE).exists()


# ──────────────────────────────────────────────
# Execution providers
# ──────────────────────────────────────────────

def test_providers_cpu_only_when_gpu_disabled():
    with patch.object(face_engine.settings, "enable_gpu", False):
        assert face_engine._providers() == ["CPUExecutionProvider"]


def test_providers_prefers_cuda_with_memory_cap():
    import onnxruntime as ort

    with patch.object(face_engine.settings, "enable_gpu", True), \
         patch.object(face_engine.settings, "insightface_gpu_mem_limit_mb", 512), \
         patch.object(ort, "get_available_providers",
                      return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), \
         patch.object(ort, "preload_dlls", side_effect=OSError("no wheel"), create=True):
        providers = face_engine._providers()

    name, options = providers[0]
    assert name == "CUDAExecutionProvider"
    assert options["gpu_mem_limit"] == 512 * 1024 * 1024
    assert providers[1] == "CPUExecutionProvider"


def test_providers_falls_back_to_cpu_without_cuda_build():
    import onnxruntime as ort

    with patch.object(face_engine.settings, "enable_gpu", True), \
         patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
        assert face_engine._providers() == ["CPUExecutionProvider"]


# ──────────────────────────────────────────────
# Engine with mocked ONNX sessions
# ──────────────────────────────────────────────

def _session(outputs):
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock(name="input")]
    sess.get_providers.return_value = ["CPUExecutionProvider"]
    sess.run.side_effect = lambda names, feed: outputs(feed)
    return sess


def _scrfd_outputs(size: int, hits: dict):
    """
    Fake SCRFD outputs. ``hits`` maps (stride, anchor_index) to
    (score, box distances[4], landmark offsets[10]) in stride units.
    """
    scores, boxes, kps = [], [], []
    for stride in InsightFaceEngine._STRIDES:
        n = (size // stride) ** 2 * InsightFaceEngine._ANCHORS_PER_CELL
        s, b, k = np.zeros((n, 1), np.float32), np.zeros((n, 4), np.float32), np.zeros((n, 10), np.float32)
        for (hit_stride, idx), (score, dist, offs) in hits.items():
            if hit_stride == stride:
                s[idx], b[idx], k[idx] = score, dist, offs
        scores.append(s), boxes.append(b), kps.append(k)
    return scores + boxes + kps


@pytest.fixture
def engine_factory(tmp_path):
    _write_models(tmp_path)

    def make(det_outputs, rec_output=None):
        det = _session(lambda feed: det_outputs)
        rec = _session(lambda feed: [rec_output if rec_output is not None else np.ones((1, 512), np.float32)])
        sessions = {DETECTOR_FILE: det, RECOGNIZER_FILE: rec}
        with patch("onnxruntime.InferenceSession", side_effect=lambda path, providers: sessions[Path(path).name]), \
             patch.object(face_engine.settings, "enable_gpu", False):
            return InsightFaceEngine(tmp_path, det_size=64, det_threshold=0.5), det, rec

    return make


def test_detect_decodes_boxes_and_landmarks_into_source_pixels(engine_factory):
    # Stride 8 on a 64×64 canvas: anchor 2*(row*8+col)+a. Pick cell (row=2, col=3).
    idx = 2 * (2 * 8 + 3)
    center = np.array([3 * 8, 2 * 8], dtype=np.float32)
    dist = np.array([1, 1, 2, 2], dtype=np.float32)
    offs = np.arange(10, dtype=np.float32) / 10
    outputs = _scrfd_outputs(64, {(8, idx): (0.9, dist, offs), (8, idx + 40): (0.3, dist, offs)})

    engine, det, _ = engine_factory(outputs)
    # 128×64 image letterboxes into 64×32 → coordinates scale by 2.
    faces = engine.detect(np.zeros((64, 128, 3), np.uint8))

    assert len(faces) == 1                       # the 0.3 score is below threshold
    face = faces[0]
    assert face.score == pytest.approx(0.9)
    expected_box = np.array([center[0] - 8, center[1] - 8, center[0] + 16, center[1] + 16]) * 2
    np.testing.assert_allclose(face.bbox, expected_box)
    expected_kps = (center + offs.reshape(5, 2) * 8) * 2
    np.testing.assert_allclose(face.landmarks, expected_kps, rtol=1e-6)
    assert face.area == pytest.approx(48 * 48)

    blob = det.run.call_args.args[1][engine._det_input]
    assert blob.shape == (1, 3, 64, 64)


def test_detect_returns_nothing_below_threshold(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    assert engine.detect(np.zeros((40, 40, 3), np.uint8)) == []


def test_detect_letterboxes_portrait_frames(engine_factory):
    engine, det, _ = engine_factory(_scrfd_outputs(64, {}))
    engine.detect(np.full((128, 32, 3), 200, np.uint8))
    blob = det.run.call_args.args[1][engine._det_input]
    # Content occupies the left 16 columns; the rest is black padding,
    # i.e. (0 − 127.5) / 128 after normalisation.
    assert np.all(blob[0, :, :, 20:] == pytest.approx(-127.5 / 128))
    assert np.all(blob[0, :, :, :16] > 0)


def test_detect_without_upscale_keeps_small_images_at_native_size(engine_factory):
    engine, det, _ = engine_factory(_scrfd_outputs(64, {}))
    engine.detect(np.full((16, 32, 3), 200, np.uint8), upscale=False)
    blob = det.run.call_args.args[1][engine._det_input]
    assert np.all(blob[0, :, :16, :32] > 0)
    assert np.all(blob[0, :, 16:, :] == pytest.approx(-127.5 / 128))
    assert np.all(blob[0, :, :, 32:] == pytest.approx(-127.5 / 128))


def test_detect_size_overrides_detector_input(engine_factory):
    engine, det, _ = engine_factory(_scrfd_outputs(64, {}))
    engine.detect(np.zeros((40, 40, 3), np.uint8), size=128)
    assert det.run.call_args.args[1][engine._det_input].shape == (1, 3, 128, 128)


# ──────────────────────────────────────────────
# Primary-face selection and detection fallbacks
# ──────────────────────────────────────────────

def _face(x1, y1, x2, y2, score=0.9):
    return face_engine.DetectedFace(
        np.array([x1, y1, x2, y2], dtype=np.float64), score, np.full((5, 2), (x1 + x2) / 2.0)
    )


def test_select_primary_face_prefers_the_fully_visible_face():
    # LFW failure case: a bystander cut off at the left edge has the bigger
    # box, but most of it lies outside the image.
    subject = _face(83, 61, 174, 186)
    bystander = _face(-29, 70, 69, 207)
    assert bystander.area > subject.area
    assert face_engine.select_primary_face([bystander, subject], (250, 250, 3)) is subject


def test_select_primary_face_still_prefers_the_larger_visible_face():
    near, far = _face(10, 10, 110, 110), _face(150, 10, 190, 50)
    assert face_engine.select_primary_face([far, near], (250, 250, 3)) is near
    assert face_engine.select_primary_face([], (250, 250, 3)) is None


def test_select_primary_face_prefers_the_centred_face_for_enrollment():
    subject = _face(100, 80, 150, 140)        # centred, small
    bystander = _face(170, 20, 250, 120)      # larger, fully visible, off to the side
    shape = (250, 250, 3)
    assert face_engine.select_primary_face([subject, bystander], shape) is bystander
    assert face_engine.select_primary_face([subject, bystander], shape, prefer_center=True) is subject


def test_detect_primary_uses_standard_pass_when_it_finds_a_face(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    face = _face(5, 5, 30, 30)
    with patch.object(engine, "detect", return_value=[face]) as detect:
        assert engine.detect_primary(np.zeros((40, 40, 3), np.uint8)) == (face, "standard")
    detect.assert_called_once()


def test_detect_primary_passes_the_selection_rule_through(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    subject, bystander = _face(100, 80, 150, 140), _face(170, 20, 250, 120)
    img = np.zeros((250, 250, 3), np.uint8)
    with patch.object(engine, "detect", return_value=[subject, bystander]):
        assert engine.detect_primary(img)[0] is bystander
        assert engine.detect_primary(img, prefer_center=True)[0] is subject


def test_detect_primary_retries_small_images_at_native_size(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    face = _face(5, 5, 30, 30)
    with patch.object(engine, "detect", side_effect=[[], [face]]) as detect:
        assert engine.detect_primary(np.zeros((40, 40, 3), np.uint8)) == (face, "native")
    assert detect.call_args.kwargs == {"upscale": False}


def test_detect_primary_retries_large_frames_at_double_size(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    face = _face(5, 5, 30, 30)
    with patch.object(engine, "detect", side_effect=[[], [face]]) as detect:
        assert engine.detect_primary(np.zeros((90, 200, 3), np.uint8)) == (face, "large")
    assert detect.call_args.kwargs == {"size": 128}


def test_detect_primary_pads_as_last_resort_and_maps_back(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    img = np.zeros((40, 50, 3), np.uint8)
    in_padded = _face(20, 15, 60, 50)
    with patch.object(engine, "detect", side_effect=[[], [], [in_padded]]) as detect:
        face, route = engine.detect_primary(img)

    assert route == "padded"
    padded = detect.call_args.args[0]
    assert padded.shape[:2] == (40 + 2 * 12, 50 + 2 * 15)
    np.testing.assert_allclose(face.bbox, [5, 3, 45, 38])
    np.testing.assert_allclose(face.landmarks, in_padded.landmarks - [15, 12])


# Upright landmarks (eyes, nose, mouth corners) and how they sit on the same
# 40 x 60 image (h x w) turned 1-3 quarter turns clockwise.
_UPRIGHT_KPS = np.array([[22, 14], [38, 14], [30, 22], [24, 30], [36, 30]], dtype=np.float64)


def _turn_points(pts, turns, h, w):
    """Where (x, y) on an h x w image lands after ``turns`` clockwise quarter turns."""
    for _ in range(turns):
        pts = np.stack([h - 1 - pts[:, 1], pts[:, 0]], axis=1)
        h, w = w, h
    return pts


def _face_at(kps, score=0.9):
    return face_engine.DetectedFace(
        np.concatenate([kps.min(axis=0) - 6, kps.max(axis=0) + 6]), score, kps
    )


@pytest.mark.parametrize("turned_by, needed", [(0, 0), (1, 3), (2, 2), (3, 1)])
def test_quarter_turns_to_upright_undoes_a_turn(turned_by, needed):
    kps = _turn_points(_UPRIGHT_KPS, turned_by, 40, 60)
    assert face_engine.quarter_turns_to_upright(kps) == needed


def test_quarter_turns_to_upright_tolerates_tilt_up_to_45_degrees():
    a = np.radians(40)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    centre = _UPRIGHT_KPS.mean(axis=0)
    assert face_engine.quarter_turns_to_upright((_UPRIGHT_KPS - centre) @ rot.T + centre) == 0


def test_quarter_turns_to_upright_needs_eyes_and_mouth_to_agree():
    # Profile view: the eyes almost coincide, so the eye line points anywhere.
    profile = _UPRIGHT_KPS.copy()
    profile[1] = profile[0] + [0.2, -1.0]
    assert face_engine.quarter_turns_to_upright(profile) == 0
    assert face_engine.quarter_turns_to_upright(np.full((5, 2), 7.0)) == 0


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_unturned_maps_a_face_back_onto_the_original_image(turns):
    h, w = 40, 60
    original = _face_at(_UPRIGHT_KPS)
    turned_kps = _turn_points(_UPRIGHT_KPS, turns, h, w)
    back = _face_at(turned_kps).unturned(turns, (h, w))
    np.testing.assert_allclose(back.landmarks, original.landmarks)
    np.testing.assert_allclose(back.bbox, original.bbox)


def test_detect_primary_finds_a_sideways_face_again_upright(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    img = np.zeros((40, 60, 3), np.uint8)
    # The photo was stored a quarter turn anticlockwise: its face lies sideways.
    sideways_kps = _turn_points(_UPRIGHT_KPS, 3, 40, 60)
    sideways_img = np.zeros((60, 40, 3), np.uint8)
    rough = _face_at(sideways_kps + [[1.5, -2.0], [-2.0, 1.0], [0, 2.5], [2.0, 0], [-1.0, -1.5]])
    with patch.object(engine, "detect", side_effect=[[rough], [_face_at(_UPRIGHT_KPS)]]) as detect:
        face, route = engine.detect_primary(sideways_img)

    assert route == "standard+rot90"
    assert detect.call_args.args[0].shape == img.shape  # searched again on the copy turned upright
    np.testing.assert_allclose(face.landmarks, sideways_kps)  # in the stored photo's coordinates
    assert face_engine.quarter_turns_to_upright(face.landmarks) == 1


def test_detect_primary_keeps_a_sideways_face_the_turned_copy_does_not_confirm(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    sideways_img = np.zeros((60, 40, 3), np.uint8)
    sideways = _face_at(_turn_points(_UPRIGHT_KPS, 3, 40, 60))
    elsewhere = _face_at(_UPRIGHT_KPS + [0, 30])  # another (upright) face, not the same one
    still_sideways = _face_at(_turn_points(_UPRIGHT_KPS, 1, 40, 60))
    for found_on_turned in ([], [elsewhere], [still_sideways]):
        with patch.object(engine, "detect", side_effect=[[sideways], found_on_turned, [], []]):
            assert engine.detect_primary(sideways_img) == (sideways, "standard")


def test_detect_primary_does_not_turn_an_upright_face(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    face = _face_at(_UPRIGHT_KPS)
    with patch.object(engine, "detect", return_value=[face]) as detect:
        assert engine.detect_primary(np.zeros((40, 60, 3), np.uint8)) == (face, "standard")
    detect.assert_called_once()


def test_detect_primary_reports_no_face(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    with patch.object(engine, "detect", return_value=[]):
        assert engine.detect_primary(np.zeros((40, 40, 3), np.uint8)) == (None, "none")


def test_align_produces_template_sized_crop(engine_factory):
    engine, _, _ = engine_factory(_scrfd_outputs(64, {}))
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    aligned = engine.align(img, ARCFACE_TEMPLATE * 1.5 + 20)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_embed_returns_unit_vector_with_arcface_normalisation(engine_factory):
    raw = np.arange(512, dtype=np.float32).reshape(1, 512)
    engine, _, rec = engine_factory(_scrfd_outputs(64, {}), rec_output=raw)

    face = np.zeros((112, 112, 3), np.uint8)
    face[..., 0] = 255                           # blue in BGR
    emb = engine.embed(face)

    assert emb.shape == (512,)
    assert np.linalg.norm(emb) == pytest.approx(1.0)
    blob = rec.run.call_args.args[1][engine._rec_input]
    # swapRB → blue lands in the last channel; (255 − 127.5) / 127.5 == 1.
    assert blob[0, 2].max() == pytest.approx(1.0)
    assert blob[0, 0].max() == pytest.approx(-1.0)


def test_get_engine_is_a_singleton():
    sentinel = MagicMock()
    with patch.object(face_engine, "_engine", None), \
         patch.object(face_engine, "InsightFaceEngine", return_value=sentinel) as cls:
        assert face_engine.get_engine() is sentinel
        assert face_engine.get_engine() is sentinel
    cls.assert_called_once()


# ──────────────────────────────────────────────
# Real weights (opt-in)
# ──────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("INSIGHTFACE_TEST_MODEL_DIR"),
    reason="set INSIGHTFACE_TEST_MODEL_DIR to run against real buffalo_l weights",
)
def test_real_models_load_and_embed_blank_frame():
    with patch.object(face_engine.settings, "enable_gpu", False):
        engine = InsightFaceEngine(Path(os.environ["INSIGHTFACE_TEST_MODEL_DIR"]))
    frame = np.zeros((480, 640, 3), np.uint8)
    assert engine.detect(frame) == []
    emb = engine.embed(cv2.resize(frame, (112, 112)))
    assert emb.shape == (512,)
