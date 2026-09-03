"""Pytest setup: make the pipeline scripts importable as top-level modules.

The scripts do ``from config import ...`` (top-level module), so the project directory must be
on ``sys.path`` before any test imports them.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Tests must never touch the server-side secrets directory or generate secrets there.
os.environ.setdefault("CTG_SECRETS_DIR", tempfile.mkdtemp(prefix="ctg-test-secrets-"))
os.environ.setdefault("CTG_BABYID_SALT", "unit-test-salt")
os.environ.setdefault("CTG_TIMESHIFT_SECRET", "unit-test-timeshift-secret")
