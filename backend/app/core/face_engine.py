"""
InsightFace engine — ONNX Runtime wrapper for embedding version 4.

Runs the InsightFace ``buffalo_l`` pack directly on ONNX Runtime, without the
``insightface`` Python package (which needs a C++ toolchain and drags in
scikit-image, albumentations, …):

  Stage 1 — DETECT   : SCRFD-10G → bounding boxes + 5 facial landmarks
  Stage 2 — ALIGN    : 5-point similarity transform onto the ArcFace template
  Stage 3 — NORMALIZE: 112×112 crop, (pixel − 127.5) / 127.5, RGB
  Stage 4 — REPRESENT: ArcFace R50 (WebFace600K) → 512-d, L2-normalised

The alignment is the exact template the recognition model was trained on, so
none of the DeepFace recipe's enhancement steps (gamma, CLAHE, denoise, …) are
applied — they would push the input away from the training distribution.

Licence: the InsightFace code is MIT, but the pretrained ``buffalo_l`` weights
are released for **non-commercial research use only**. A commercial
deployment needs a licence from the InsightFace authors.
"""

import hashlib
import logging
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

DETECTOR_FILE = "det_10g.onnx"
RECOGNIZER_FILE = "w600k_r50.onnx"
MODEL_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
# Pinned so a tampered or truncated download is never loaded.
MODEL_PACK_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"

# ArcFace canonical landmark template for a 112×112 crop:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_NMS_THRESHOLD = 0.4
# Border added on every side, as a fraction of the image, by the last
# detection fallback (see InsightFaceEngine.detect_primary).
_PAD_FRACTION = 0.3


@dataclass
class DetectedFace:
    bbox: np.ndarray       # (4,) x1, y1, x2, y2 in source-image pixels
    score: float           # detector confidence, 0–1
    landmarks: np.ndarray  # (5, 2) in source-image pixels

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1]))

    def visible_area(self, image_shape: tuple) -> float:
        """Area of the part of the box that lies inside the image."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, min(x2, w) - max(x1, 0.0)) * max(0.0, min(y2, h) - max(y1, 0.0)))

    def shifted(self, dx: float, dy: float) -> "DetectedFace":
        return DetectedFace(
            self.bbox + np.array([dx, dy, dx, dy]), self.score, self.landmarks + np.array([dx, dy])
        )


def select_primary_face(
    faces: list[DetectedFace], image_shape: tuple, prefer_center: bool = False
) -> Optional[DetectedFace]:
    """
    The face to embed when the image holds several.

    Default (search): visible area, scaled down by the share of the box that
    falls outside the image, so a face cut off at the border loses to a fully
    visible one wherever the subject stands — CCTV subjects are rarely centred.

    ``prefer_center`` (enrollment): InsightFace's own rule, box area minus
    twice the squared distance from the image centre — the subject of a
    mugshot or ID photo is the centred face.

    Plain box area — the previous rule — counts the part of the box outside
    the image, so a bystander half out of shot often outranked the subject: on
    LFW it scored 97.1% against 99.7% for these rules (backend/scripts/
    benchmark_engines.py --lfw-pairs).
    """
    if not faces:
        return None
    if prefer_center:
        h, w = image_shape[:2]

        def score(f: DetectedFace) -> float:
            cx, cy = (f.bbox[0] + f.bbox[2]) / 2 - w / 2, (f.bbox[1] + f.bbox[3]) / 2 - h / 2
            return f.area - 2.0 * (cx * cx + cy * cy)
    else:
        def score(f: DetectedFace) -> float:
            return f.visible_area(image_shape) ** 2 / max(f.area, 1e-6)
    return max(faces, key=score)


# ---------- Model files ----------
def ensure_model_files(model_dir: Path) -> None:
    """
    Make sure the detector and recogniser ONNX files exist in ``model_dir``.

    Downloads the pinned ``buffalo_l`` pack only when files are missing and
    ``INSIGHTFACE_AUTO_DOWNLOAD`` is enabled; only the two files this engine
    uses are extracted.
    """
    needed = [DETECTOR_FILE, RECOGNIZER_FILE]
    missing = [name for name in needed if not (model_dir / name).is_file()]
    if not missing:
        return
    if not settings.insightface_auto_download:
        raise FileNotFoundError(
            f"InsightFace model files {missing} not found in {model_dir} and "
            "INSIGHTFACE_AUTO_DOWNLOAD is disabled"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading InsightFace model pack to %s", model_dir)
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        archive = Path(tmp) / "pack.zip"
        with urllib.request.urlopen(MODEL_PACK_URL, timeout=60) as resp, open(archive, "wb") as out:
            shutil.copyfileobj(resp, out)

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != MODEL_PACK_SHA256:
            raise RuntimeError(f"InsightFace model pack checksum mismatch: {digest}")

        with zipfile.ZipFile(archive) as zf:
            for name in missing:
                member = next(n for n in zf.namelist() if Path(n).name == name)
                with zf.open(member) as src, open(Path(tmp) / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                # Rename is atomic, so a crash never leaves a half-written model.
                (Path(tmp) / name).replace(model_dir / name)


# ---------- ONNX Runtime session setup ----------
def _providers() -> list:
    import onnxruntime as ort

    providers: list = ["CPUExecutionProvider"]
    if settings.enable_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        # The TF base image ships cuDNN 8; ORT's CUDA 12 build needs cuDNN 9,
        # which comes from the nvidia-cudnn-cu12 wheel. Different sonames, so
        # both coexist; preload makes ORT find the wheel's copy.
        try:
            ort.preload_dlls(cuda=False, cudnn=True, msvc=False)
        except Exception as exc:  # noqa: BLE001 — falls back to system libs
            logger.debug("onnxruntime.preload_dlls failed: %s", exc)
        providers.insert(0, (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "gpu_mem_limit": settings.insightface_gpu_mem_limit_mb * 1024 * 1024,
                # Grow the arena by what is requested rather than doubling,
                # so the VRAM footprint stays close to the real working set.
                "arena_extend_strategy": "kSameAsRequested",
            },
        ))
    return providers


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity transform (rotation, uniform scale, translation)
    mapping ``src`` points onto ``dst`` — Umeyama's method, the same estimate
    InsightFace's ``norm_crop`` uses. Returns a 2×3 affine matrix.
    """
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = (s * d).sum() / src_c.var(axis=0).sum()

    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = scale * rot
    m[:, 2] = dst_mean - scale * rot @ src_mean
    return m


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


class InsightFaceEngine:
    """SCRFD detector + ArcFace recogniser on ONNX Runtime. Thread-safe."""

    _STRIDES = (8, 16, 32)
    _ANCHORS_PER_CELL = 2

    def __init__(self, model_dir: Path, det_size: int = 640, det_threshold: float = 0.5):
        import onnxruntime as ort

        ensure_model_files(model_dir)
        providers = _providers()
        self._det = ort.InferenceSession(str(model_dir / DETECTOR_FILE), providers=providers)
        self._rec = ort.InferenceSession(str(model_dir / RECOGNIZER_FILE), providers=providers)
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name
        self.det_size = det_size
        self.det_threshold = det_threshold
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._anchor_lock = threading.Lock()
        logger.info(
            "InsightFace engine ready (providers=%s). Pretrained weights are "
            "licensed for non-commercial research use only.",
            self._det.get_providers(),
        )

    @property
    def providers(self) -> list[str]:
        return self._det.get_providers()

    # ── Stage 1: DETECT ──────────────────────────────────────────
    def _anchors(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        with self._anchor_lock:
            if key not in self._anchor_cache:
                centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                centers = np.repeat(centers, self._ANCHORS_PER_CELL, axis=0)
                self._anchor_cache[key] = centers
            return self._anchor_cache[key]

    def detect(
        self, img_bgr: np.ndarray, size: Optional[int] = None, upscale: bool = True
    ) -> list[DetectedFace]:
        """
        Return every face scoring above ``det_threshold``, best first.

        ``size`` overrides the square detector input (default ``det_size``).
        With ``upscale=False`` an image smaller than the input is placed at its
        native size instead of being enlarged to fill it.
        """
        import cv2

        # Letterbox into a square canvas, keeping the aspect ratio.
        h, w = img_bgr.shape[:2]
        size = size or self.det_size
        if not upscale and max(h, w) <= size:
            new_h, new_w = h, w
        elif h > w:
            new_h, new_w = size, max(1, int(size * w / h))
        else:
            new_h, new_w = max(1, int(size * h / w)), size
        scale = new_h / h
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img_bgr if (new_h, new_w) == (h, w) else cv2.resize(img_bgr, (new_w, new_h))

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True
        )
        outputs = self._det.run(None, {self._det_input: blob})

        # Outputs come grouped: scores ×3 strides, boxes ×3, landmarks ×3.
        n = len(self._STRIDES)
        all_scores, all_boxes, all_kps = [], [], []
        for i, stride in enumerate(self._STRIDES):
            scores = outputs[i].reshape(-1)
            boxes = outputs[i + n].reshape(-1, 4) * stride
            kps = outputs[i + 2 * n].reshape(-1, 10) * stride
            centers = self._anchors(size // stride, size // stride, stride)

            keep = np.where(scores >= self.det_threshold)[0]
            if keep.size == 0:
                continue
            c = centers[keep]
            b = boxes[keep]
            all_scores.append(scores[keep])
            all_boxes.append(np.stack(
                [c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]],
                axis=-1,
            ))
            k = kps[keep].reshape(-1, 5, 2)
            all_kps.append(c[:, None, :] + k)

        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale
        keep = _nms(boxes, scores, _NMS_THRESHOLD)
        return [DetectedFace(boxes[i], float(scores[i]), kps[i]) for i in keep]

    def detect_primary(
        self, img_bgr: np.ndarray, prefer_center: bool = False
    ) -> tuple[Optional[DetectedFace], str]:
        """
        The one face to embed (see ``select_primary_face``; pass
        ``prefer_center`` for enrollment photos) and the pass that found it:
        "standard" | "native" | "large" | "padded" | "none".

        The standard pass runs first, so an image it already handled keeps the
        same detections. The fallbacks run only when it finds nothing:

          native — an image smaller than the detector input, at its own size.
                   Enlarged to fill the input, a tightly cropped face overflows
                   the detector's largest anchors and is missed.
          large  — a frame bigger than the input, at twice the input size, for
                   faces too small to survive the downscale (CCTV).
          padded — a black border round the image, for faces cut off at the edge.
        """
        import cv2

        def primary(faces: list[DetectedFace]) -> DetectedFace:
            return select_primary_face(faces, img_bgr.shape, prefer_center)

        faces = self.detect(img_bgr)
        if faces:
            return primary(faces), "standard"

        h, w = img_bgr.shape[:2]
        if max(h, w) < self.det_size:
            faces, route = self.detect(img_bgr, upscale=False), "native"
        elif max(h, w) > self.det_size:
            faces, route = self.detect(img_bgr, size=2 * self.det_size), "large"
        if faces:
            return primary(faces), route

        ph, pw = int(h * _PAD_FRACTION), int(w * _PAD_FRACTION)
        padded = cv2.copyMakeBorder(img_bgr, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=0)
        faces = [f.shifted(-pw, -ph) for f in self.detect(padded)]
        if faces:
            return primary(faces), "padded"
        return None, "none"

    # ── Stages 2–4: ALIGN → NORMALIZE → REPRESENT ────────────────
    def align(self, img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Warp the face onto the 112×112 ArcFace template (BGR uint8)."""
        import cv2

        m = _similarity_transform(landmarks.astype(np.float64), ARCFACE_TEMPLATE.astype(np.float64))
        return cv2.warpAffine(img_bgr, m, (112, 112), borderValue=0.0)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """512-d L2-normalised embedding of an aligned 112×112 BGR face."""
        import cv2

        blob = cv2.dnn.blobFromImage(
            aligned_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        embedding = self._rec.run(None, {self._rec_input: blob})[0][0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding


_engine: Optional[InsightFaceEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> InsightFaceEngine:
    """Process-wide engine, created on first use (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InsightFaceEngine(
                    Path(settings.insightface_model_dir).expanduser(),
                    det_size=settings.insightface_det_size,
                    det_threshold=settings.insightface_det_threshold,
                )
    return _engine
