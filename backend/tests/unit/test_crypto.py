"""Test AES-256 embedding encryption helpers."""

import base64
import secrets

import numpy as np
import pytest

from app.core.crypto import _decode_key, decrypt_embedding_bytes, encrypt_embedding_vector


def test_encrypt_decrypt_roundtrip():
    embedding = np.random.randn(512).astype(np.float32)
    encrypted = encrypt_embedding_vector(embedding)
    assert isinstance(encrypted, bytes)

    decrypted = decrypt_embedding_bytes(encrypted)
    assert decrypted.shape[0] == 512
    assert np.allclose(decrypted, embedding, atol=1e-6)


def test_decrypt_rejects_short_payload():
    with pytest.raises(ValueError):
        decrypt_embedding_bytes(b"too-short")


@pytest.mark.parametrize("raw, message", [
    ("", "must be set"),
    ("   ", "must be set"),
    ("not base64 at all!", "or 64 hex"),
    (base64.b64encode(bytes(16)).decode(), "must decode to 32 bytes"),
    # 64 characters but not hex: decoded as base64 (48 bytes), so still rejected.
    (base64.b64encode(b"\xff" * 48).decode(), "must decode to 32 bytes"),
])
def test_bad_encryption_key_is_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        _decode_key(raw)


def test_hex_and_base64_keys_are_accepted():
    key = secrets.token_bytes(32)
    assert _decode_key(key.hex()) == key
    assert _decode_key(f"  {base64.b64encode(key).decode()}\n") == key


def test_decrypt_rejects_unexpected_dimensions():
    payload = encrypt_embedding_vector(np.ones(128, np.float32))
    with pytest.raises(ValueError, match="unexpected dimensions"):
        decrypt_embedding_bytes(payload)
