"""Pytest setup: make the pipeline scripts importable as top-level modules.

The scripts do ``from config import ...`` (top-level module), so the project directory must be
on ``sys.path`` before any test imports them.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
