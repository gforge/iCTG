"""Pseudonymous identifiers derived from the BabyID salt.

``MotherID`` lets downstream code group the pregnancies of one mother (mother-level
train/validation/test splits, leakage exclusions) without exposing the CTG ``PatientID``.
It is ``sha256(salt | "mother" | PatientID)`` truncated to 16 hex characters, computed
identically in DuckDB (stage 7) and Python (stage 8) so both outputs agree.
"""

from __future__ import annotations

import hashlib

MOTHER_TAG = "mother"
MOTHER_ID_LENGTH = 16


def mother_id(salt: str, patient_id: str) -> str:
    digest = hashlib.sha256(f"{salt}|{MOTHER_TAG}|{patient_id}".encode()).hexdigest()
    return digest[:MOTHER_ID_LENGTH]


def mother_id_sql(salt: str, patient_sql: str) -> str:
    """DuckDB expression equal to ``mother_id(salt, <patient_sql>)``."""
    safe_salt = salt.replace("'", "''")
    return (
        f"substr(sha256(concat('{safe_salt}', '|{MOTHER_TAG}|', CAST({patient_sql} AS VARCHAR))), "
        f"1, {MOTHER_ID_LENGTH})"
    )
