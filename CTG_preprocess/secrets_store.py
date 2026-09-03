"""Secrets that must never be committed: the BabyID salt and the time-shift secret.

Resolution order for a secret called ``name``:

1. environment variable ``CTG_<NAME>`` (e.g. ``CTG_BABYID_SALT``);
2. the file ``<secrets dir>/<name>`` where the directory is ``CTG_SECRETS_DIR`` or, by default,
   ``<reduction root>/secrets`` (created with mode 700, files with mode 600);
3. otherwise a new random secret is generated and written to that file, with a warning, so the
   pipeline is reproducible from then on. Keep the directory with the intermediate stage data:
   it is part of what re-identifies the outputs.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from config import DEFAULT_SECRETS_DIR

SECRET_NAMES = ("babyid_salt", "timeshift_secret")


def _env_var(name: str) -> str:
    return f"CTG_{name.upper()}"


def secrets_dir() -> Path:
    return Path(os.environ.get("CTG_SECRETS_DIR", DEFAULT_SECRETS_DIR))


def get_secret(name: str, *, create: bool = True) -> str:
    """Return the secret ``name`` (see module docstring for the resolution order).

    With ``create=False`` a missing secret raises ``FileNotFoundError`` instead of being
    generated; use that for stages that must reuse an earlier stage's secret.
    """
    if name not in SECRET_NAMES:
        raise ValueError(f"Unknown secret {name!r}; expected one of {SECRET_NAMES}")
    value = os.environ.get(_env_var(name))
    if value:
        return value.strip()

    path = secrets_dir() / name
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"Secret file is empty: {path}")
        return value
    if not create:
        raise FileNotFoundError(f"Secret {name!r} not found: set {_env_var(name)} or create {path}")

    value = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    print(
        f"WARNING: generated a new secret {name!r} at {path}. Keep this file: BabyIDs and "
        "time shifts are only reproducible with it. Set "
        f"{_env_var(name)} to reuse an existing secret instead.",
        file=sys.stderr,
    )
    return value
