"""
AES-256-GCM helpers for embedding encryption at rest.
"""

import base64
import binascii
import os
from functools import lru_cache

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings


def _decode_key(raw: str) -> bytes:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("DB_ENCRYPTION_KEY must be set")

    if len(raw) == 64:
        try:
            return bytes.fromhex(raw)
        except ValueError:
            pass

    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("DB_ENCRYPTION_KEY must be 32 bytes (base64) or 64 hex") from exc

    if len(key) != 32:
        raise ValueError("DB_ENCRYPTION_KEY must decode to 32 bytes")

    return key


@lru_cache
def _aesgcm() -> AESGCM:
    return AESGCM(_decode_key(settings.db_encryption_key))


def encrypt_embedding_vector(embedding: np.ndarray) -> bytes:
    data = embedding.astype(np.float32).tobytes()
    nonce = os.urandom(12)
    cipher = _aesgcm().encrypt(nonce, data, None)
    return nonce + cipher


def decrypt_embedding_bytes(payload: bytes, expected_dim: int = 512) -> np.ndarray:
    if len(payload) < 13:
        raise ValueError("Encrypted embedding payload is too short")

    nonce = payload[:12]
    cipher = payload[12:]
    data = _aesgcm().decrypt(nonce, cipher, None)

    embedding = np.frombuffer(data, dtype=np.float32)
    if embedding.size != expected_dim:
        raise ValueError("Decrypted embedding has unexpected dimensions")

    return embedding
