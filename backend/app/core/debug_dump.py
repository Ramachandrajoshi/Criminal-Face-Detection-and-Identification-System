"""
Debug face dumps — write the face image after every pipeline stage to disk.

**Debug only. Off by default (DEBUG_FACE_DUMP=false).** This persists raw
face images, which AGENTS.md §15 forbids in normal operation; enable it only
on a local/research deployment while diagnosing matching problems (e.g.
same-person distances drifting under different lighting), then switch it off
and delete the folder.

Layout — one folder per pipeline run under ``DEBUG_FACE_DUMP_DIR``::

    debug_faces/
      20261024-101502-123456_search_a1b2c3/
        00_input.jpg              original upload, byte-for-byte
        v3_01_detect.png          frame with the chosen box drawn
        v3_02_crop.png            detector's aligned crop
        v3_03_uint8.png … v3_09_final_112.png   each _preprocess_for_arcface step
        v4_01_detect.png          SCRFD box + 5 landmarks
        v4_02_aligned_112.png     what the ArcFace R50 actually sees
        manifest.json             per-stage lighting stats + params + result

Every image in the manifest carries ``mean`` / ``std`` luminance, the share of
near-black and near-white pixels and a Laplacian sharpness score, so two runs
of the same person under different lighting can be compared numerically as
well as visually. Matches are recorded by face id and distance only.

The active run lives in a ContextVar. ``run_inference`` copies the context
into the inference pool, so stage functions reach the run without any change
to their signatures. Every write is best-effort: a dump failure is logged and
never affects the pipeline result.
"""

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class FaceDump:
    """One pipeline run's dump folder and manifest. Thread-safe."""

    def __init__(self, root: Path, flow: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        self.dir = root / f"{stamp}_{flow}_{uuid.uuid4().hex[:6]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._manifest: dict = {"flow": flow, "started_utc": stamp, "settings": _recipe_settings(), "stages": []}

    def save_bytes(self, name: str, data: bytes) -> None:
        path = self.dir / f"{name}.{_guess_ext(data)}"
        path.write_bytes(data)
        with self._lock:
            self._manifest["stages"].append({"file": path.name, "bytes": len(data)})

    def save_image(self, name: str, image: np.ndarray, rgb: bool, **meta) -> None:
        import cv2

        img = _to_uint8(image)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb and img.ndim == 3 else img
        path = self.dir / f"{name}.png"  # lossless, so the dump adds no artefacts
        cv2.imwrite(str(path), bgr)
        entry = {"file": path.name, "shape": list(img.shape), **_lighting_stats(bgr), **meta}
        with self._lock:
            self._manifest["stages"].append(entry)

    def note(self, key: str, value) -> None:
        with self._lock:
            self._manifest[key] = value

    def write_manifest(self) -> None:
        with self._lock:
            self._manifest["stages"].sort(key=lambda s: s["file"])
            text = json.dumps(self._manifest, indent=2, default=_json_default)
        (self.dir / "manifest.json").write_text(text, encoding="utf-8")


_current: ContextVar[Optional[FaceDump]] = ContextVar("face_dump", default=None)


@contextmanager
def face_dump(flow: str, image_bytes: Optional[bytes] = None) -> Iterator[Optional[FaceDump]]:
    """
    Open a dump folder for one pipeline run (no-op when DEBUG_FACE_DUMP is off).

    Stage helpers called inside the block, including on the inference pool via
    ``run_inference``, write into this run. The manifest is written on exit,
    also when the pipeline raises.
    """
    if not settings.debug_face_dump:
        yield None
        return
    try:
        dump = FaceDump(Path(settings.debug_face_dump_dir).expanduser(), flow)
    except Exception as exc:  # noqa: BLE001 — debugging must never break the pipeline
        logger.warning("Face dump disabled for this run: %s", exc)
        yield None
        return

    token = _current.set(dump)
    try:
        if image_bytes is not None:
            _safe(dump.save_bytes, "00_input", image_bytes)
        yield dump
    finally:
        _current.reset(token)
        _safe(dump.write_manifest)
        logger.info("Face dump written to %s", dump.dir)


def dump_stage(name: str, image: np.ndarray, rgb: bool = False, **meta) -> None:
    """Save ``image`` as stage ``name`` of the active run, if there is one."""
    dump = _current.get()
    if dump is not None:
        _safe(dump.save_image, name, image, rgb, **meta)


def dump_detection(name: str, frame_bgr: np.ndarray, bbox, landmarks=None, label: str = "", **meta) -> None:
    """Save the full frame with the chosen box (x1, y1, x2, y2) and landmarks drawn on it."""
    dump = _current.get()
    if dump is None:
        return

    def draw() -> None:
        import cv2

        canvas = _to_uint8(frame_bgr).copy()
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        thick = max(2, canvas.shape[1] // 400)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), thick)
        if landmarks is not None:
            for x, y in np.asarray(landmarks).reshape(-1, 2):
                cv2.circle(canvas, (int(x), int(y)), thick + 1, (0, 0, 255), -1)
        if label:
            cv2.putText(canvas, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * thick, (0, 255, 0), thick)
        # Stats describe the face region, not the whole frame.
        h, w = canvas.shape[:2]
        crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        face_stats = _lighting_stats(_to_uint8(crop)) if crop.size else {}
        dump.save_image(name, canvas, rgb=False, bbox=[x1, y1, x2, y2],
                        face_region=face_stats, **meta)

    _safe(draw)


def dump_note(key: str, value) -> None:
    """Record a value (e.g. the match result) in the active run's manifest."""
    dump = _current.get()
    if dump is not None:
        dump.note(key, value)


# ---------- helpers ----------
def _safe(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Face dump write failed: %s", exc)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    # DeepFace crops are float32 in [0, 1].
    scale = 255.0 if float(np.max(image, initial=0.0)) <= 1.0 else 1.0
    return (image.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)


def _lighting_stats(img_bgr: np.ndarray) -> dict:
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return {
        "lum_mean": round(float(gray.mean()), 1),
        "lum_std": round(float(gray.std()), 1),   # global contrast
        "pct_dark": round(float((gray <= 25).mean() * 100), 1),
        "pct_bright": round(float((gray >= 230).mean() * 100), 1),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
    }


def _recipe_settings() -> dict:
    keys = (
        "embedding_version", "deepface_detector", "deepface_model", "match_threshold",
        "match_threshold_insightface", "arcface_normalization", "enable_upscale_small_crops",
        "small_crop_min_side", "enable_denoising", "enable_gamma_correction", "enable_clahe",
        "clahe_clip_limit", "enable_unsharp_mask", "unsharp_strength",
        "insightface_det_size", "insightface_det_threshold",
        "match_threshold_adaface", "adaface_det_threshold",
    )
    return {k: getattr(settings, k) for k in keys}


def _guess_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "img"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
