"""
Liveness / Anti-spoofing module.

Currently delegates to deepface's built-in anti_spoofing flag.
In production, this should be expanded with dedicated liveness checks
(e.g., face-3d-assistant, ORL, or custom CNN).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def check_liveness(image_bytes: bytes) -> bool:
    """
    Perform a basic liveness check on the uploaded image.

    Returns True if the image passes liveness (real person),
    False if a spoof (photo, video, mask) is detected.

    This is a placeholder — production should integrate a
    dedicated anti-spoofing model.
    """
    try:
        import cv2
        from deepface import DeepFace

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return False

        # DeepFace's extract_faces has anti_spoofing parameter
        from deepface import DeepFace

        # Use a simple heuristic: check if the face is detected
        # In production, use a dedicated liveness model
        result = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True,
        )

        if not result:
            return False

        # Placeholder: assume real if face detected
        # TODO: Integrate a dedicated liveness model
        return True

    except Exception as exc:
        logger.error("Liveness check error: %s", exc)
        return False


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

        result = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True,
            anti_spoofing=True,
        )

        if not result:
            return {"is_live": False, "spoof_probability": 1.0}

        # DeepFace may return is_real in the result
        face = result[0]
        is_real = face.get("is_real", True)
        spoof_prob = 1.0 - face.get("real_score", 1.0)

        return {
            "is_live": is_real,
            "spoof_probability": float(spoof_prob),
        }

    except Exception as exc:
        logger.error("DeepFace liveness error: %s", exc)
        return {"is_live": False, "spoof_probability": 1.0}
