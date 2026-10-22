"""
Head-pose estimation from the 5 SCRFD landmarks.

The InsightFace detector already returns left eye, right eye, nose tip and the
two mouth corners for every face, so pose costs nothing extra to compute. The
estimate is approximate (roughly ±10–15°), which is enough to bucket a face as
frontal / three-quarter / profile and to reject faces turned too far to embed
reliably. It is not a substitute for a 3D landmark model.

Geometry
--------
1. Roll — the angle of the line between the eyes. All points are rotated by
   −roll about the eye midpoint so the eyes are level before measuring yaw
   and pitch.
2. Yaw — the nose tip sits ~0.5 inter-eye distances in front of the face's
   midline. Turning the head by θ moves its projection sideways by
   0.5·tan(θ) eye-distances, so θ = atan(2 · offset).
3. Pitch — the same model vertically: the nose tip's position between the
   eye line and the mouth line, relative to the frontal ratio of the ArcFace
   template.

Sign conventions (image space, as the camera sees the face)
  yaw   > 0 — nose to the right of the face midline (face turned image-right)
  pitch > 0 — looking up
  roll  > 0 — head tilted clockwise in the image
"""

import math
from dataclasses import dataclass

import numpy as np

# Nose-tip depth in front of the face plane, in units of inter-eye distance
# (yaw) and eye-to-mouth distance (pitch).
_NOSE_DEPTH_RATIO = 0.5

# Nose position between eye line (0) and mouth line (1) on a frontal face,
# taken from the ArcFace alignment template.
_FRONTAL_NOSE_RATIO = (71.7366 - 51.5989) / (92.2848 - 51.5989)

FRONTAL_MAX_YAW = 15.0
THREE_QUARTER_MAX_YAW = 45.0
PITCH_BUCKET_DEG = 20.0


@dataclass(frozen=True)
class HeadPose:
    yaw: float    # degrees
    pitch: float  # degrees
    roll: float   # degrees

    @property
    def bucket(self) -> str:
        return pose_bucket(self.yaw, self.pitch)

    @property
    def deviation(self) -> float:
        """How far from frontal — used to choose the most frontal image."""
        return abs(self.yaw) + abs(self.pitch)


def estimate_pose(landmarks: np.ndarray) -> HeadPose:
    """Estimate yaw / pitch / roll from (5, 2) SCRFD landmarks in pixels."""
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    left_eye, right_eye, nose, mouth_left, mouth_right = pts

    eye_vec = right_eye - left_eye
    eye_dist = float(np.hypot(*eye_vec))
    if eye_dist < 1e-6:
        return HeadPose(0.0, 0.0, 0.0)
    roll = math.atan2(eye_vec[1], eye_vec[0])

    # Level the eyes: rotate every point by −roll about the eye midpoint.
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    derotate = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    eye_mid = (left_eye + right_eye) / 2.0
    nose_x, nose_y = derotate @ (nose - eye_mid)
    mouth_x, mouth_y = derotate @ ((mouth_left + mouth_right) / 2.0 - eye_mid)

    # The eye midpoint is at x=0 after rotation; average it with the mouth
    # midpoint so one badly placed landmark does not swing the midline.
    midline_x = mouth_x / 2.0
    yaw = math.atan((nose_x - midline_x) / eye_dist / _NOSE_DEPTH_RATIO)

    if mouth_y > 1e-6:
        nose_ratio = nose_y / mouth_y
        pitch = math.atan((_FRONTAL_NOSE_RATIO - nose_ratio) / _NOSE_DEPTH_RATIO)
    else:
        pitch = 0.0

    return HeadPose(math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def pose_bucket(yaw: float, pitch: float) -> str:
    """
    Coarse pose label, e.g. ``frontal``, ``left_3q``, ``right_profile_down``.

    Left / right are image-space, matching the sign of ``yaw``.
    """
    side = "right" if yaw > 0 else "left"
    abs_yaw = abs(yaw)
    if abs_yaw < FRONTAL_MAX_YAW:
        horizontal = "frontal"
    elif abs_yaw < THREE_QUARTER_MAX_YAW:
        horizontal = f"{side}_3q"
    else:
        horizontal = f"{side}_profile"

    if pitch >= PITCH_BUCKET_DEG:
        return f"{horizontal}_up"
    if pitch <= -PITCH_BUCKET_DEG:
        return f"{horizontal}_down"
    return horizontal
