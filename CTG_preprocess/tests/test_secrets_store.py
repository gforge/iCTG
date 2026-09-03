"""Secrets resolution: env var, file, generation (never the committed default)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import secrets_store


def test_env_var_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CTG_SECRETS_DIR", str(tmp_path))
    monkeypatch.setenv("CTG_BABYID_SALT", " from-env ")
    assert secrets_store.get_secret("babyid_salt") == "from-env"
    assert not (tmp_path / "babyid_salt").exists()


def test_file_is_used_then_generated_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CTG_SECRETS_DIR", str(tmp_path / "secrets"))
    monkeypatch.delenv("CTG_TIMESHIFT_SECRET", raising=False)
    with pytest.raises(FileNotFoundError):
        secrets_store.get_secret("timeshift_secret", create=False)
    first = secrets_store.get_secret("timeshift_secret")
    path = tmp_path / "secrets" / "timeshift_secret"
    assert path.exists()
    assert len(first) >= 32
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(os.stat(path.parent).st_mode & 0o777) == "0o700"
    assert secrets_store.get_secret("timeshift_secret") == first
    path.write_text("explicit\n")
    assert secrets_store.get_secret("timeshift_secret") == "explicit"


def test_unknown_secret_rejected() -> None:
    with pytest.raises(ValueError):
        secrets_store.get_secret("password")
