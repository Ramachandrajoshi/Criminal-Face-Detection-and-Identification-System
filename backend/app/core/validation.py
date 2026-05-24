"""Input validation helpers."""

import io
from typing import Tuple

from PIL import Image


def validate_image_dimensions(image_bytes: bytes, min_size: int = 64) -> Tuple[int, int]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            width, height = img.size
    except Exception as exc:
        raise ValueError("Invalid image data") from exc

    if width < min_size or height < min_size:
        raise ValueError(f"Image dimensions must be at least {min_size}x{min_size}")

    return width, height
