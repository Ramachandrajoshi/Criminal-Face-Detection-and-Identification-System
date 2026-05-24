"""Test AES-256 embedding encryption helpers."""

import numpy as np
import pytest

from app.core.crypto import decrypt_embedding_bytes, encrypt_embedding_vector


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
