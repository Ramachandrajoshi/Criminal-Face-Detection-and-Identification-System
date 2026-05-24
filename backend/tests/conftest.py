import base64
import os
import secrets
import sys
from pathlib import Path

from passlib.context import CryptContext


def _ensure_test_env() -> None:
    os.environ.setdefault("JWT_SECRET", secrets.token_urlsafe(32))
    os.environ.setdefault("POSTGRES_PASSWORD", secrets.token_urlsafe(16))
    os.environ.setdefault(
        "DB_ENCRYPTION_KEY",
        base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
    )

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    test_username = f"admin_{secrets.token_hex(4)}"
    test_password = secrets.token_urlsafe(16)
    os.environ["ADMIN_USERNAME"] = test_username
    os.environ["ADMIN_PASSWORD_HASH"] = pwd_context.hash(test_password)
    os.environ["ADMIN_TEST_USERNAME"] = test_username
    os.environ["ADMIN_TEST_PASSWORD"] = test_password


_ensure_test_env()

# Ensure the project root is on sys.path for test collection
# /app/tests/conftest.py → parent.parent = /app
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
