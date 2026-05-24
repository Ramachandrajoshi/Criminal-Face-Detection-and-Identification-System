"""Test image dimension validation helper."""

import io

import pytest
from PIL import Image

from app.core.validation import validate_image_dimensions


def _make_image_bytes(size=(112, 112)) -> bytes:
    img = Image.new("RGB", size, color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_image_dimensions_ok():
    data = _make_image_bytes((120, 80))
    width, height = validate_image_dimensions(data, min_size=64)
    assert width == 120
    assert height == 80


def test_validate_image_dimensions_rejects_invalid():
    with pytest.raises(ValueError):
        validate_image_dimensions(b"not-an-image", min_size=64)


def test_validate_image_dimensions_rejects_small():
    data = _make_image_bytes((32, 32))
    with pytest.raises(ValueError):
        validate_image_dimensions(data, min_size=64)
