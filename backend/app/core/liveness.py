"""
Liveness / Anti-spoofing module.

Currently delegates to deepface's built-in anti_spoofing flag.
In production, this should be expanded with dedicated liveness checks
(e.g., face-3d-assistant, ORL, or custom CNN).
"""

import logging
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def check_liveness(image_bytes: bytes) -> bool:
    """Return True if the image passes anti-spoofing checks."""
    result = check_liveness_with_deepface(image_bytes)
    return result.get("is_live", False)


def check_liveness_with_deepface(image_bytes: bytes) -> dict:
    """
    Use DeepFace's anti_spoofing flag for liveness detection.

    Returns:
        {"is_live": bool, "spoof_probability": float}
    """
    try:
        import cv2
        import numpy as np

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"is_live": False, "spoof_probability": 1.0}

        from deepface import DeepFace

        verify_result = DeepFace.verify(
            img1_path=frame,
            img2_path=frame,
            model_name=settings.deepface_model,
            detector_backend=settings.deepface_detector,
            distance_metric="cosine",
            enforce_detection=True,
            anti_spoofing=True,
        )

        is_real = verify_result.get("is_real")
        real_score = verify_result.get("real_score")

        verified_flag = verify_result.get("verified")

        if is_real is None and real_score is None and verified_flag is not None:
            is_real = bool(verified_flag)
            real_score = 1.0 if is_real else 0.0

        if is_real is None and real_score is None:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=settings.deepface_detector,
                enforce_detection=True,
                align=True,
                anti_spoofing=True,
            )
            if not faces:
                return {"is_live": False, "spoof_probability": 1.0}

            face = faces[0]
            is_real = face.get("is_real", True)
            real_score = face.get("real_score", 1.0)

        if is_real is None:
            is_real = bool(verify_result.get("verified", False))

        if real_score is None:
            real_score = 1.0 if is_real else 0.0

        spoof_prob = 1.0 - float(real_score) if is_real else 1.0

        return {
            "is_live": bool(is_real),
            "spoof_probability": float(spoof_prob),
        }

    except Exception as exc:
        logger.error("DeepFace liveness error: %s", exc)
        return {"is_live": False, "spoof_probability": 1.0}


def analyze_video_liveness(
    video_bytes: bytes,
    suffix: str = ".mp4",
    max_frames: int = 6,
    min_duration_sec: float = 2.5,
    max_duration_sec: float = 5.5,
) -> dict:
    """
    Sample frames spread across a short video clip and run DeepFace
    anti-spoofing on each, aggregating a single liveness verdict.

    Scoring multiple frames spread across the clip (rather than one still) is
    materially harder to defeat with a printed photo or a video/photo replayed
    on a phone/monitor: a presentation attack has to fool the anti-spoofing
    model on *every* sampled frame, not just get lucky once on a single shot.

    Returns a dict with keys:
        is_live            — bool, aggregate verdict (majority of sampled
                              frames scored live)
        spoof_probability  — float, mean spoof probability across analyzed
                              frames
        frames_analyzed    — int, number of frames actually scored
        duration_sec       — float, decoded clip duration
        error              — present only on early-exit: "invalid_video"
                              (unreadable/corrupt/no decodable frames) or
                              "duration_out_of_range"
    """
    import os
    import tempfile

    import cv2

    tmp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {
                "is_live": False, "spoof_probability": 1.0,
                "frames_analyzed": 0, "duration_sec": 0.0,
                "error": "invalid_video",
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / fps) if fps > 0 else 0.0

        if total_frames <= 0 or fps <= 0:
            return {
                "is_live": False, "spoof_probability": 1.0,
                "frames_analyzed": 0, "duration_sec": duration_sec,
                "error": "invalid_video",
            }

        if not (min_duration_sec <= duration_sec <= max_duration_sec):
            return {
                "is_live": False, "spoof_probability": 1.0,
                "frames_analyzed": 0, "duration_sec": duration_sec,
                "error": "duration_out_of_range",
            }

        # Evenly spaced sample indices across the middle 80% of the clip,
        # skipping the first/last 10% to dodge capture start/stop artifacts
        # (autofocus hunting, motion blur from raising the phone/device).
        n = max(1, min(max_frames, total_frames))
        lo, hi = int(total_frames * 0.1), max(int(total_frames * 0.9), 1)
        hi = max(hi, lo + 1)
        step = (hi - lo) / max(n - 1, 1) if n > 1 else 0
        indices = sorted({int(lo + i * step) for i in range(n)})

        spoof_scores = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ok2, buf = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            result = check_liveness_with_deepface(buf.tobytes())
            spoof_scores.append(result["spoof_probability"])

        if not spoof_scores:
            return {
                "is_live": False, "spoof_probability": 1.0,
                "frames_analyzed": 0, "duration_sec": duration_sec,
                "error": "invalid_video",
            }

        avg_spoof = sum(spoof_scores) / len(spoof_scores)
        live_count = sum(1 for s in spoof_scores if s < 0.5)
        is_live = (live_count / len(spoof_scores)) > 0.5

        return {
            "is_live": is_live,
            "spoof_probability": avg_spoof,
            "frames_analyzed": len(spoof_scores),
            "duration_sec": duration_sec,
        }

    except Exception as exc:
        logger.error("Video liveness error: %s", exc)
        return {
            "is_live": False, "spoof_probability": 1.0,
            "frames_analyzed": 0, "duration_sec": 0.0,
            "error": "invalid_video",
        }
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
