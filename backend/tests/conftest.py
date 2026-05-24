import sys
from pathlib import Path

# Ensure the project root is on sys.path for test collection
# /app/tests/conftest.py → parent.parent = /app
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
