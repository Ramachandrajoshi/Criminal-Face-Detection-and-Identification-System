"""
AdaFace engine — ONNX Runtime wrapper for embedding version 5.

Version 4 (app/core/face_engine.py) with both models swapped for ones whose
weights allow commercial use. Face choice, the no-face fallbacks, the second
detection of sideways faces and the alignment are inherited unchanged:

  Stage 1 — DETECT   : YOLOv5-Face M → bounding boxes + 5 facial landmarks
  Stage 2 — ALIGN    : 5-point similarity transform onto the ArcFace template
  Stage 3 — NORMALIZE: 112×112 crop, (pixel − 127.5) / 127.5, BGR
  Stage 4 — REPRESENT: AdaFace IR-101 (WebFace12M) → 512-d, L2-normalised

Chosen over RetinaFace ResNet-34/50 on the regression data in
backend/testdata (results/candidates_2026-10-30.json): it is the most
accurate on CCTV-sized faces and level with version 4 on clean photos.

Licences: AdaFace code and weights are MIT (github.com/mk-minchul/AdaFace,
ONNX export by github.com/yakhyo/adaface-onnx). YOLOv5-Face code and weights
are GPL-3.0 (github.com/deepcam-cn/yolov5-face, ONNX export by
github.com/yakhyo/yolov5-face-onnx-inference); the weights are downloaded at
run time, not shipped with this code. The training sets (WIDER FACE,
WebFace12M) are released for non-commercial research; whether those terms
reach users of the trained weights is unsettled, so take legal advice before
a commercial deployment.
"""

import hashlib
import logging
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.face_engine import (
    _NMS_THRESHOLD,
    DetectedFace,
    InsightFaceEngine,
    _nms,
    _providers,
)

logger = logging.getLogger(__name__)

DETECTOR_FILE = "yolov5m_face.onnx"
RECOGNIZER_FILE = "adaface_ir_101.onnx"
# file → (URL, SHA-256). Pinned so a tampered or truncated download is never loaded.
MODEL_FILES = {
    DETECTOR_FILE: (
        "https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5m_face.onnx",
        "04302ce27a15bde3e20945691b688e2dd018a10e92dd8932146bede6a49207b2",
    ),
    RECOGNIZER_FILE: (
        "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx",
        "f2eb07d03de0af560a82e1214df799fec5e09375d43521e2868f9dc387e5a43e",
    ),
}


# ---------- Model files ----------
def ensure_model_files(model_dir: Path) -> None:
    """
    Make sure the detector and recogniser ONNX files exist in ``model_dir``.

    Missing files are downloaded only when ``ADAFACE_AUTO_DOWNLOAD`` is
    enabled, and kept only if their checksum matches.
    """
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if not missing:
        return
    if not settings.adaface_auto_download:
        raise FileNotFoundError(
            f"AdaFace model files {missing} not found in {model_dir} and "
            "ADAFACE_AUTO_DOWNLOAD is disabled"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        for name in missing:
            url, sha256 = MODEL_FILES[name]
            logger.info("Downloading %s to %s", name, model_dir)
            part = Path(tmp) / name
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != sha256:
                raise RuntimeError(f"AdaFace model file {name} checksum mismatch: {digest.hexdigest()}")
            # Rename is atomic, so a crash never leaves a half-written model.
            part.replace(model_dir / name)


class AdaFaceEngine(InsightFaceEngine):
    """YOLOv5-Face detector + AdaFace recogniser on ONNX Runtime. Thread-safe."""

    # The YOLOv5-Face ONNX input is fixed at 640×640.
    INPUT_SIZE = 640
    # A pass larger than the input (the "large" fallback for small CCTV faces)
    # runs as 640 px tiles this far apart, so neighbouring tiles overlap by
    # 160 px and a small face is always whole in one of them.
    _TILE_STRIDE = 480
    _PAD_VALUE = 114

    def __init__(self, model_dir: Path, det_threshold: float = 0.5):
        import onnxruntime as ort

        ensure_model_files(model_dir)
        providers = _providers()
        self._det = ort.InferenceSession(str(model_dir / DETECTOR_FILE), providers=providers)
        self._rec = ort.InferenceSession(str(model_dir / RECOGNIZER_FILE), providers=providers)
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name
        self.det_size = self.INPUT_SIZE
        self.det_threshold = det_threshold
        logger.info("AdaFace engine ready (providers=%s)", self._det.get_providers())

    # ── Stage 1: DETECT ──────────────────────────────────────────
    def _detect_canvas(self, img_bgr: np.ndarray, scale: float):
        """
        Faces in an image no larger than the input once scaled by ``scale``,
        centred on a grey canvas (the model's training letterbox). Returns
        boxes (N, 4), scores (N,), landmarks (N, 5, 2) in ``img_bgr`` pixels.
        """
        import cv2

        h, w = img_bgr.shape[:2]
        new_h, new_w = (h, w) if scale == 1.0 else (int(h * scale), int(w * scale))
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), self._PAD_VALUE, dtype=np.uint8)
        top, left = (self.INPUT_SIZE - new_h) // 2, (self.INPUT_SIZE - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = img_bgr if scale == 1.0 else cv2.resize(img_bgr, (new_w, new_h))

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (self.INPUT_SIZE, self.INPUT_SIZE), (0, 0, 0), swapRB=True)
        # Rows: centre x, centre y, width, height, face score, 5 landmarks (x, y), class score.
        pred = self._det.run(None, {self._det_input: blob})[0][0]
        pred = pred[pred[:, 4] >= self.det_threshold]
        half = pred[:, 2:4] / 2
        boxes = np.concatenate([pred[:, :2] - half, pred[:, :2] + half], axis=1)
        kps = pred[:, 5:15].reshape(-1, 5, 2)
        offset = np.array([left, top], dtype=np.float32)
        return (boxes - np.tile(offset, 2)) / scale, pred[:, 4], (kps - offset) / scale

    def detect(
        self, img_bgr: np.ndarray, size: Optional[int] = None, upscale: bool = True
    ) -> list[DetectedFace]:
        """
        Return every face scoring above ``det_threshold``, best first.

        Same contract as ``InsightFaceEngine.detect``: ``size`` is the square
        the image is fitted into, ``upscale=False`` keeps a small image at its
        native size. A ``size`` above the fixed 640 input is detected as
        overlapping 640 px tiles of the image fitted into ``size``.
        """
        import cv2

        size = size or self.det_size
        h, w = img_bgr.shape[:2]
        scale = 1.0 if (not upscale and max(h, w) <= size) else min(size / h, size / w)
        if size <= self.INPUT_SIZE:
            boxes, scores, kps = self._detect_canvas(img_bgr, scale)
        else:
            big = img_bgr if scale == 1.0 else cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            big_h, big_w = big.shape[:2]

            def starts(n: int) -> list[int]:
                last = max(n - self.INPUT_SIZE, 0)
                return sorted({*range(0, last + 1, self._TILE_STRIDE), last})

            parts = []
            for y in starts(big_h):
                for x in starts(big_w):
                    b, s, k = self._detect_canvas(big[y:y + self.INPUT_SIZE, x:x + self.INPUT_SIZE], 1.0)
                    parts.append(((b + [x, y, x, y]) / scale, s, (k + [x, y]) / scale))
            boxes, scores, kps = (np.concatenate(p) for p in zip(*parts))

        if scores.size == 0:
            return []
        keep = _nms(boxes, scores, _NMS_THRESHOLD)
        return [DetectedFace(boxes[i], float(scores[i]), kps[i]) for i in keep]

    # ── Stage 4: REPRESENT ───────────────────────────────────────
    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """512-d L2-normalised embedding of an aligned 112×112 BGR face (AdaFace takes BGR)."""
        import cv2

        blob = cv2.dnn.blobFromImage(
            aligned_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False
        )
        embedding = self._rec.run(None, {self._rec_input: blob})[0][0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding


_engine: Optional[AdaFaceEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> AdaFaceEngine:
    """Process-wide engine, created on first use (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AdaFaceEngine(
                    Path(settings.adaface_model_dir).expanduser(),
                    det_threshold=settings.adaface_det_threshold,
                )
    return _engine
