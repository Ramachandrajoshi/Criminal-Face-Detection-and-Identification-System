"""
Test liveness / anti-spoofing detection.

Covers:
- check_liveness: basic heuristic
- check_liveness_with_deepface: DeepFace anti_spoofing flag
- Spoof-blocked paths in the pipeline
- Invalid image handling
"""

from unittest.mock import AsyncMock, MagicMock, patch
import io
import sys
import asyncio

import numpy as np
import pytest
from PIL import Image
from fastapi import UploadFile

# Create a mock deepface module if not available
_mock_installed = False
if "deepface" not in sys.modules:
    import types

    _deepface_mock = types.ModuleType("deepface")

    class _DeepFaceMock:
        @staticmethod
        def extract_faces(*args, **kwargs):
            return []

        @staticmethod
        def represent(*args, **kwargs):
            return []

        @staticmethod
        def verify(*args, **kwargs):
            return {}

    _deepface_mock.DeepFace = _DeepFaceMock
    sys.modules["deepface"] = _deepface_mock
    sys.modules["deepface.api"] = types.ModuleType("deepface.api")
    sys.modules["deepface.api"].DeepFace = _DeepFaceMock
    _mock_installed = True


def make_test_image(color="red"):
    """Return raw image bytes of a test image."""
    img = Image.new("RGB", (112, 112), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_invalid_image_bytes():
    """Return bytes that are NOT a valid image."""
    return b"this is not an image at all \x00\x01\x02\x03"


class TestLivenessCheck:
    """Test basic liveness detection (heuristic path)."""

    def test_liveness_returns_true_for_valid_image(self):
        mock_verify = MagicMock(return_value={"is_real": True, "real_score": 0.95})
        with patch("deepface.DeepFace.verify", mock_verify):
            from app.core.liveness import check_liveness
            result = check_liveness(make_test_image())
            assert result is True

    def test_liveness_returns_false_for_no_face(self):
        mock_verify = MagicMock(return_value={"is_real": False, "real_score": 0.1})
        with patch("deepface.DeepFace.verify", mock_verify):
            from app.core.liveness import check_liveness
            result = check_liveness(make_test_image())
            assert result is False

    def test_liveness_returns_false_for_invalid_bytes(self):
        """Non-image bytes should fail to decode → False."""
        with patch("deepface.DeepFace.verify", MagicMock(return_value={})): 
            from app.core.liveness import check_liveness
            result = check_liveness(make_invalid_image_bytes())
            assert result is False

    def test_liveness_returns_false_on_exception(self):
        """Exceptions in liveness check should return False."""
        with patch("deepface.DeepFace.verify", side_effect=Exception("boom")):
            from app.core.liveness import check_liveness
            result = check_liveness(make_test_image())
            assert result is False

    def test_liveness_with_deepface_detector_backend(self):
        """Verify that retinaface backend is used."""
        with patch("deepface.DeepFace.verify") as mock_verify:
            from app.core.liveness import check_liveness
            check_liveness(make_test_image())
            mock_verify.assert_called_once()
            call_kwargs = mock_verify.call_args[1]
            assert call_kwargs.get("detector_backend") == "retinaface"


class TestLivenessDeepFace:
    """Test DeepFace anti-spoofing liveness."""

    def test_live_face_detected(self):
        mock_verify = MagicMock(return_value={"is_real": True, "real_score": 0.92})
        with patch("deepface.DeepFace.verify", mock_verify):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_test_image())
            assert result["is_live"] is True
            assert result["spoof_probability"] < 0.5

    def test_spoof_detected(self):
        mock_verify = MagicMock(return_value={"is_real": False, "real_score": 0.15})
        with patch("deepface.DeepFace.verify", mock_verify):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_test_image())
            assert result["is_live"] is False
            assert result["spoof_probability"] > 0.5

    def test_no_face_returns_spoof(self):
        mock_verify = MagicMock(return_value={})
        with patch("deepface.DeepFace.verify", mock_verify), \
             patch("deepface.DeepFace.extract_faces", MagicMock(return_value=[])):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_test_image())
            assert result["is_live"] is False
            assert result["spoof_probability"] == 1.0

    def test_invalid_image_returns_spoof(self):
        """Invalid image bytes should be treated as spoof."""
        with patch("deepface.DeepFace.verify", MagicMock(return_value={})), \
             patch("deepface.DeepFace.extract_faces", MagicMock(return_value=[])):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_invalid_image_bytes())
            assert result["is_live"] is False
            assert result["spoof_probability"] == 1.0

    def test_low_real_score_is_spoof(self):
        """Real score < 0.5 should indicate spoof."""
        mock_verify = MagicMock(return_value={"is_real": True, "real_score": 0.3})
        with patch("deepface.DeepFace.verify", mock_verify):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_test_image())
            # is_real is True but real_score is low → spoof_probability should be high
            assert result["spoof_probability"] > 0.5

    def test_missing_real_score_defaults_to_live(self):
        """If is_real/real_score keys are missing, default to live."""
        mock_verify = MagicMock(return_value={"verified": True})
        with patch("deepface.DeepFace.verify", mock_verify):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_test_image())
            assert result["is_live"] is True
            assert result["spoof_probability"] < 0.5

    def test_deepface_exception_returns_spoof(self):
        """Exception during DeepFace liveness check should return spoof."""
        with patch("deepface.DeepFace.verify", side_effect=Exception("model load fail")):
            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(make_test_image())
            assert result["is_live"] is False
            assert result["spoof_probability"] == 1.0


class TestSpoofBlockedPath:
    """Test that pipeline returns SPOOF_BLOCKED when liveness fails."""

    def test_spoof_blocked_triggers_audit_entry(self):
        """Spoof detection should call add_audit_entry with SPOOF_BLOCKED."""
        with patch("app.core.liveness.check_liveness_with_deepface") as mock_live:
            mock_live.return_value = {"is_live": False, "spoof_probability": 0.95}

            from app.core.liveness import check_liveness_with_deepface
            result = check_liveness_with_deepface(b"fake")
            assert result["is_live"] is False

    def test_pipeline_calls_liveness_on_search(self):
        """run_pipeline should call liveness check only when enforce_liveness=True."""
        liveness_called = []

        def liveness_side_effect(data):
            liveness_called.append(data)
            return True

        mock_detect = MagicMock(return_value=None)

        with patch("app.core.pipeline.check_liveness_on_bytes", side_effect=liveness_side_effect), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock()):

            from app.core.pipeline import run_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image()),
            )

            # 1. When enforce_liveness is True
            asyncio.run(run_pipeline(file, AsyncMock(), enforce_liveness=True))
            assert len(liveness_called) == 1

            # 2. When enforce_liveness is False (default)
            liveness_called.clear()
            file.file.seek(0)
            asyncio.run(run_pipeline(file, AsyncMock(), enforce_liveness=False))
            assert len(liveness_called) == 0

    def test_pipeline_calls_liveness_on_register(self):
        """register_pipeline should never call liveness check."""
        liveness_called = []

        def liveness_side_effect(data):
            liveness_called.append(data)
            return True

        mock_detect = MagicMock(return_value=None)

        with patch("app.core.pipeline.check_liveness_on_bytes", side_effect=liveness_side_effect), \
             patch("app.core.pipeline.detect_face", mock_detect), \
             patch("app.db.vector_ops.add_audit_entry", AsyncMock()):

            from app.core.pipeline import register_pipeline
            file = UploadFile(
                filename="test.jpg",
                file=io.BytesIO(make_test_image()),
            )

            asyncio.run(register_pipeline(file, AsyncMock(), "Test"))

            assert len(liveness_called) == 0


class _FakeVideoCapture:
    """Stand-in for cv2.VideoCapture, driven entirely by constructor args."""

    def __init__(self, fps=10.0, total_frames=40, opens=True):
        self._fps = fps
        self._total_frames = total_frames
        self._opens = opens

    def isOpened(self):
        return self._opens

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._total_frames
        return 0

    def set(self, prop, value):
        pass

    def read(self):
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        pass


class TestVideoLiveness:
    """Test analyze_video_liveness: multi-frame video anti-spoofing."""

    def test_video_within_duration_majority_live_is_live(self):
        """~4s clip where most sampled frames score live → is_live True."""
        fake_cap = _FakeVideoCapture(fps=10.0, total_frames=40)  # 4.0s
        spoof_scores = iter([0.1, 0.2, 0.1, 0.3, 0.1, 0.2])

        with patch("cv2.VideoCapture", return_value=fake_cap), \
             patch("app.core.liveness.check_liveness_with_deepface",
                   side_effect=lambda b: {"is_live": True, "spoof_probability": next(spoof_scores)}):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"fake video bytes")

        assert result["is_live"] is True
        assert "error" not in result
        assert result["frames_analyzed"] > 0
        assert result["spoof_probability"] < 0.5

    def test_video_within_duration_majority_spoof_is_not_live(self):
        """~4s clip where most sampled frames score spoof → is_live False."""
        fake_cap = _FakeVideoCapture(fps=10.0, total_frames=40)  # 4.0s

        with patch("cv2.VideoCapture", return_value=fake_cap), \
             patch("app.core.liveness.check_liveness_with_deepface",
                   return_value={"is_live": False, "spoof_probability": 0.95}):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"fake video bytes")

        assert result["is_live"] is False
        assert result["spoof_probability"] > 0.5

    def test_video_too_short_is_rejected(self):
        """Clip shorter than min_duration_sec should short-circuit with an error."""
        fake_cap = _FakeVideoCapture(fps=10.0, total_frames=10)  # 1.0s

        with patch("cv2.VideoCapture", return_value=fake_cap):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"fake video bytes")

        assert result["is_live"] is False
        assert result["error"] == "duration_out_of_range"
        assert result["frames_analyzed"] == 0

    def test_video_too_long_is_rejected(self):
        """Clip longer than max_duration_sec should short-circuit with an error."""
        fake_cap = _FakeVideoCapture(fps=10.0, total_frames=100)  # 10.0s

        with patch("cv2.VideoCapture", return_value=fake_cap):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"fake video bytes")

        assert result["is_live"] is False
        assert result["error"] == "duration_out_of_range"

    def test_unopenable_video_is_invalid(self):
        """A corrupt/undecodable file should return an invalid_video error."""
        fake_cap = _FakeVideoCapture(opens=False)

        with patch("cv2.VideoCapture", return_value=fake_cap):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"not a video")

        assert result["is_live"] is False
        assert result["error"] == "invalid_video"
        assert result["frames_analyzed"] == 0

    def test_zero_frame_count_is_invalid(self):
        """A video with no decodable frames/fps metadata is invalid."""
        fake_cap = _FakeVideoCapture(fps=0.0, total_frames=0)

        with patch("cv2.VideoCapture", return_value=fake_cap):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"fake video bytes")

        assert result["is_live"] is False
        assert result["error"] == "invalid_video"

    def test_video_exception_returns_spoof(self):
        """Any unexpected exception during analysis should fail closed."""
        with patch("cv2.VideoCapture", side_effect=Exception("boom")):
            from app.core.liveness import analyze_video_liveness
            result = analyze_video_liveness(b"fake video bytes")

        assert result["is_live"] is False
        assert result["spoof_probability"] == 1.0
        assert result["error"] == "invalid_video"

