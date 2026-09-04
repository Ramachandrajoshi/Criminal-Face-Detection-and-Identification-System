"""
Liveness check endpoint — standalone anti-spoofing verification.

``POST /api/v1/liveness/video`` samples several frames spread across a short
video clip and runs DeepFace anti-spoofing on each, aggregating a single
verdict. Read-only: no database writes, no audit trail — it does not
detect/match/register a face, it only scores "real vs. spoof" for the clip
handed to it.

A single-frame endpoint was tried first and removed: DeepFace's anti-spoofing
model reliably scored genuine live captures as spoofed too (relying on a
single still starves it of the depth/motion cues it needs), so it produced
no usable signal. Sampling multiple frames across a short clip gives the
model enough to work with.

This is intentionally separate from ``POST /api/v1/search``'s
``is_live_capture`` flag, which already gates a *search* on liveness and
logs SPOOF_BLOCKED to the audit trail. Use this endpoint when you want a
liveness verdict on its own — e.g. a pre-flight check before capture, or a
standalone check from another flow — without running the full
detect → embed → match pipeline or writing to the audit log.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.auth import get_current_user
from app.core.inference_executor import run_inference
from app.core.liveness import analyze_video_liveness
from app.schemas.face import VideoLivenessResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["liveness"])

ALLOWED_VIDEO_MIME = {"video/mp4", "video/webm", "video/quicktime"}
_VIDEO_SUFFIX_BY_MIME = {"video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov"}
MAX_VIDEO_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_VIDEO_DURATION_SEC = 2.5
MAX_VIDEO_DURATION_SEC = 5.5
MAX_VIDEO_FRAMES = 6


@router.post("/liveness/video", response_model=VideoLivenessResponse)
async def check_liveness_video(
    file: UploadFile = File(
        ...,
        description=(
            f"Short face video, {MIN_VIDEO_DURATION_SEC:.1f}-{MAX_VIDEO_DURATION_SEC:.1f}s "
            "(MP4/WebM/QuickTime, ≤ 5 MB)"
        ),
    ),
    _user: dict = Depends(get_current_user),
):
    """
    Run anti-spoofing across several frames sampled from a short video clip.

    A presentation attack (printed photo, or a phone/monitor replaying a
    photo or video) has to fool the anti-spoofing model on every sampled
    frame, not just one lucky shot.

    Requirements:
    - Container: MP4, WebM, or QuickTime/MOV.
    - Size: ≤ 5 MB.
    - Duration: 2.5-5.5 seconds (a small tolerance is allowed around the
      nominal 3-5s window to absorb encoder rounding).

    Does not touch the database: no audit entry, no face match.
    """
    if not file.content_type or file.content_type not in ALLOWED_VIDEO_MIME:
        raise HTTPException(status_code=400, detail="Only MP4/WebM/QuickTime videos are accepted")

    content = await file.read()
    if len(content) > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Video must be ≤ 5 MB")

    suffix = _VIDEO_SUFFIX_BY_MIME.get(file.content_type, ".mp4")
    result = await run_inference(
        analyze_video_liveness,
        content,
        suffix=suffix,
        max_frames=MAX_VIDEO_FRAMES,
        min_duration_sec=MIN_VIDEO_DURATION_SEC,
        max_duration_sec=MAX_VIDEO_DURATION_SEC,
    )

    error = result.get("error")
    if error == "duration_out_of_range":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video duration must be between {MIN_VIDEO_DURATION_SEC:.1f}s and "
                f"{MAX_VIDEO_DURATION_SEC:.1f}s (got {result['duration_sec']:.1f}s)"
            ),
        )
    if error == "invalid_video":
        raise HTTPException(
            status_code=400,
            detail="Could not read video — file may be corrupt, empty, or use an unsupported codec",
        )

    is_live = bool(result["is_live"])
    return VideoLivenessResponse(
        is_live=is_live,
        spoof_probability=float(result["spoof_probability"]),
        frames_analyzed=int(result["frames_analyzed"]),
        message=(
            f"Live face detected across {result['frames_analyzed']} sampled frames."
            if is_live
            else "Spoofing attempt suspected — the video looks like a photo, replayed "
                 "video, or screen capture rather than a live recording."
        ),
    )
