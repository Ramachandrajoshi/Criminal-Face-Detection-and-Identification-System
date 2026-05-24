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
