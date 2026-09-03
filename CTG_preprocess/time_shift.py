"""Stage 8: time-shift the final outputs so that absolute dates cannot be linked back.

Every pregnancy (BabyID) gets a shift of whole days that is applied to all its dates and
timestamps: the CTG rows in ``ctg_final.parquet``, the pretraining ``all_sessions`` export and
the date/timestamp columns of ``registry.csv``. Time of day and every relative quantity
(seconds before birth, day offsets in the long tables, ages) are unchanged.

Per mother (CTG ``PatientID``):

* a base shift is drawn uniformly from ``[-max_days, +max_days]``;
* her pregnancies are ordered by their true anchor date (last CTG timestamp of the final
  window) and every interval between consecutive pregnancies is multiplied by
  ``1 +/- u`` with ``u`` drawn uniformly from ``[jitter_min, jitter_max]`` and a random sign,
  so the order of births is preserved but their true spacing cannot be recovered;
* the shift of pregnancy *k* is then ``shifted_anchor_k - true_anchor_k``.

All randomness is seeded from ``sha256(secret | PatientID)`` so a rerun with the same secret
(see ``secrets_store.py``) reproduces the same shifts. BabyIDs that only occur in the
pretraining export (dropped before stage 3's main output) are shifted independently from
``sha256(secret | BabyID)``.

The per-BabyID shift table is written next to the outputs as ``timeshift_key.parquet``. It
re-identifies the dates and belongs with the intermediate stage data, not the deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import (
    DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE7_CTG_PARQUET,
    DEFAULT_STAGE7_REGISTRY_CSV,
    DEFAULT_STAGE8_ALL_SESSIONS_DIR,
    DEFAULT_STAGE8_CTG_PARQUET,
    DEFAULT_STAGE8_KEY_FILE,
    DEFAULT_STAGE8_MOTHERS_CSV,
    DEFAULT_STAGE8_REGISTRY_CSV,
    DEFAULT_TIMESHIFT_JITTER_MAX,
    DEFAULT_TIMESHIFT_JITTER_MIN,
    DEFAULT_TIMESHIFT_MAX_DAYS,
    DEFAULT_TIMESHIFT_REGISTRY_COLUMNS,
)
from pseudonyms import mother_id
from secrets_store import get_secret


def _safe(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _parquet_source(path: str | Path) -> str:
    """``read_parquet(...)`` over a file, or over the top-level parquet files of a directory.

    Stage 3 writes ``stage3_sessions_bucket_XXXX.parquet`` files into its directory; the
    ``all_sessions`` export lives in a subdirectory and is deliberately not matched.
    """
    p = Path(path)
    if p.is_dir():
        return f"read_parquet('{_safe(p)}/*.parquet', union_by_name=true)"
    return f"read_parquet('{_safe(p)}')"


def _seed(secret: str, key: str) -> int:
    digest = hashlib.sha256(f"{secret}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_shift_table(
    pregnancies: pd.DataFrame,
    secret: str,
    *,
    max_days: int = DEFAULT_TIMESHIFT_MAX_DAYS,
    jitter_min: float = DEFAULT_TIMESHIFT_JITTER_MIN,
    jitter_max: float = DEFAULT_TIMESHIFT_JITTER_MAX,
) -> pd.DataFrame:
    """Return ``BabyID, shift_days`` for every row of ``pregnancies``.

    ``pregnancies`` needs ``BabyID``, ``PatientID`` (may be null) and ``anchor`` (date-like).
    Rows without ``PatientID`` are shifted independently, keyed on their BabyID.
    """
    if not (0 <= jitter_min <= jitter_max < 1):
        raise ValueError("Need 0 <= jitter_min <= jitter_max < 1")
    if pregnancies["BabyID"].duplicated().any():
        raise ValueError("pregnancies must have one row per BabyID")

    df = pregnancies[["BabyID", "PatientID", "anchor"]].copy()
    df["anchor"] = pd.to_datetime(df["anchor"]).dt.normalize()
    df["shift_days"] = 0
    df["key"] = df["PatientID"].where(df["PatientID"].notna(), "baby:" + df["BabyID"].astype(str))
    df = df.sort_values(["key", "anchor", "BabyID"], kind="stable").reset_index(drop=True)

    keys = df["key"].to_numpy()
    anchors = df["anchor"].to_numpy()
    shifts = np.zeros(len(df), dtype=np.int64)

    start = 0
    n = len(df)
    while start < n:
        end = start
        while end < n and keys[end] == keys[start]:
            end += 1
        rng = np.random.default_rng(_seed(secret, str(keys[start])))
        base = int(rng.integers(-max_days, max_days + 1))
        shifts[start] = base
        if end - start > 1:
            true_days = (
                (anchors[start:end] - anchors[start]).astype("timedelta64[D]").astype(np.int64)
            )
            prev_true = 0
            prev_shifted = 0
            for i in range(1, end - start):
                interval = int(true_days[i] - prev_true)
                u = float(rng.uniform(jitter_min, jitter_max)) * (1 if rng.integers(0, 2) else -1)
                new_interval = int(round(interval * (1.0 + u)))
                if interval > 0:
                    new_interval = max(1, new_interval)
                shifted = prev_shifted + new_interval
                shifts[start + i] = base + shifted - int(true_days[i])
                prev_true = int(true_days[i])
                prev_shifted = shifted
        start = end

    df["shift_days"] = shifts
    return df[["BabyID", "shift_days"]].reset_index(drop=True)


def _load_pregnancies(
    con: duckdb.DuckDBPyConnection, stage3_path: Path, all_sessions_dir: Path | None
) -> pd.DataFrame:
    """One row per BabyID with the mother's PatientID and the anchor date.

    The anchor is the last CTG timestamp of the pregnancy in stage 3's main output. BabyIDs
    that only exist in the all-sessions export are added with PatientID NULL.
    """
    con.execute(
        f"""
        CREATE TEMP TABLE preg AS
        SELECT BabyID, ANY_VALUE(PatientID) AS PatientID, CAST(MAX(Timestamp) AS DATE) AS anchor
        FROM {_parquet_source(stage3_path)}
        GROUP BY BabyID
        """
    )
    if all_sessions_dir is not None and all_sessions_dir.exists():
        con.execute(
            f"""
            INSERT INTO preg
            SELECT a.BabyID, NULL, CAST(MAX(a.Timestamp) AS DATE)
            FROM {_parquet_source(all_sessions_dir)} a
            LEFT JOIN preg p USING (BabyID)
            WHERE p.BabyID IS NULL
            GROUP BY a.BabyID
            """
        )
    return con.execute("SELECT BabyID, PatientID, anchor FROM preg").df()


def _shift_parquet(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, label: str) -> int:
    """Copy ``src`` to ``dst`` with ``Timestamp`` shifted; every BabyID must have a shift."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_safe(src)}')").fetchone()
    con.execute(
        f"""
        COPY (
            SELECT s.* REPLACE (CAST(s.Timestamp + to_days(k.shift_days) AS TIMESTAMP) AS Timestamp)
            FROM read_parquet('{_safe(src)}') s
            JOIN key k USING (BabyID)
        ) TO '{_safe(dst)}' (FORMAT PARQUET)
        """
    )
    written = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_safe(dst)}')").fetchone()
    n_in = int(total[0]) if total else 0
    n_out = int(written[0]) if written else 0
    if n_in != n_out:
        raise RuntimeError(f"{label}: {n_in - n_out} rows lost, some BabyIDs have no shift")
    return n_out


def _shift_registry(
    con: duckdb.DuckDBPyConnection, src: Path, dst: Path, columns: list[str]
) -> tuple[int, list[str]]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW reg_in AS SELECT * FROM read_csv_auto('{_safe(src)}', header=true)"
    )
    types = {row[0]: row[1] for row in con.execute("DESCRIBE reg_in").fetchall()}
    shifted = [c for c in columns if c in types]
    replace = ", ".join(
        f"CAST(r.{c} + to_days(k.shift_days) AS {types[c]}) AS {c}" for c in shifted
    )
    select = f"r.* REPLACE ({replace})" if replace else "r.*"
    con.execute(
        f"""
        COPY (
            SELECT {select}
            FROM reg_in r
            JOIN key k USING (BabyID)
            ORDER BY BabyID
        ) TO '{_safe(dst)}' (HEADER, DELIMITER ',')
        """
    )
    n_in = con.execute("SELECT COUNT(*) FROM reg_in").fetchone()
    n_out = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{_safe(dst)}', header=true)"
    ).fetchone()
    a, b = int(n_in[0]) if n_in else 0, int(n_out[0]) if n_out else 0
    if a != b:
        raise RuntimeError(f"registry: {a - b} rows lost, some BabyIDs have no shift")
    return b, shifted


def time_shift_outputs(
    *,
    stage3_path: str | Path = DEFAULT_STAGE3_DIR,
    all_sessions_in: str | Path | None = DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    registry_in: str | Path = DEFAULT_STAGE7_REGISTRY_CSV,
    ctg_in: str | Path = DEFAULT_STAGE7_CTG_PARQUET,
    registry_out: str | Path = DEFAULT_STAGE8_REGISTRY_CSV,
    ctg_out: str | Path = DEFAULT_STAGE8_CTG_PARQUET,
    all_sessions_out: str | Path = DEFAULT_STAGE8_ALL_SESSIONS_DIR,
    key_out: str | Path = DEFAULT_STAGE8_KEY_FILE,
    mothers_out: str | Path | None = None,
    secret: str | None = None,
    babyid_salt: str | None = None,
    max_days: int = DEFAULT_TIMESHIFT_MAX_DAYS,
    jitter_min: float = DEFAULT_TIMESHIFT_JITTER_MIN,
    jitter_max: float = DEFAULT_TIMESHIFT_JITTER_MAX,
    registry_columns: list[str] | None = None,
) -> dict[str, object]:
    """Run stage 8 and return a summary of counts (no identifiers)."""
    stage3_path = Path(stage3_path)
    all_sessions_dir = Path(all_sessions_in) if all_sessions_in else None
    registry_in, ctg_in = Path(registry_in), Path(ctg_in)
    registry_out, ctg_out = Path(registry_out), Path(ctg_out)
    all_sessions_out_dir, key_out = Path(all_sessions_out), Path(key_out)
    columns = list(
        DEFAULT_TIMESHIFT_REGISTRY_COLUMNS if registry_columns is None else registry_columns
    )
    if secret is None:
        secret = get_secret("timeshift_secret")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    pregnancies = _load_pregnancies(con, stage3_path, all_sessions_dir)
    key = build_shift_table(
        pregnancies, secret, max_days=max_days, jitter_min=jitter_min, jitter_max=jitter_max
    )
    con.register("key_df", key)
    con.execute("CREATE TEMP TABLE key AS SELECT BabyID, shift_days FROM key_df")
    key_out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY key TO '{_safe(key_out)}' (FORMAT PARQUET)")

    # BabyID -> MotherID for every pregnancy (same hash as stage 7's registry.csv).
    if babyid_salt is None:
        babyid_salt = get_secret("babyid_salt")
    mothers_df = pregnancies[["BabyID", "PatientID"]].copy()
    mothers_df["MotherID"] = [
        mother_id(babyid_salt, str(p)) if pd.notna(p) else ""
        for p in mothers_df["PatientID"].tolist()
    ]
    # Defaults next to the shifted registry so tests and custom output dirs stay self-contained.
    mothers_out = (
        Path(mothers_out) if mothers_out is not None else registry_out.parent / "mothers.csv"
    )
    mothers_out.parent.mkdir(parents=True, exist_ok=True)
    mothers_df[["BabyID", "MotherID"]].sort_values("BabyID").to_csv(mothers_out, index=False)

    n_registry, shifted_cols = _shift_registry(con, registry_in, registry_out, columns)
    n_ctg = _shift_parquet(con, ctg_in, ctg_out, "ctg_final")

    n_all = 0
    all_files = 0
    if all_sessions_dir is not None and all_sessions_dir.exists():
        for src in sorted(all_sessions_dir.glob("*.parquet")):
            n_all += _shift_parquet(con, src, all_sessions_out_dir / src.name, src.name)
            all_files += 1

    mothers = int(pregnancies["PatientID"].nunique())
    multi = int((pregnancies.groupby("PatientID").size() > 1).sum())
    summary: dict[str, object] = {
        "pregnancies": int(len(key)),
        "mothers": mothers,
        "mothers_with_several_pregnancies": multi,
        "babyids_without_patientid": int(pregnancies["PatientID"].isna().sum()),
        "max_days": max_days,
        "jitter": [jitter_min, jitter_max],
        "shift_days_min": int(key["shift_days"].min()),
        "shift_days_max": int(key["shift_days"].max()),
        "registry_rows": n_registry,
        "registry_columns_shifted": shifted_cols,
        "ctg_rows": n_ctg,
        "all_sessions_files": all_files,
        "all_sessions_rows": n_all,
        "mothers_rows": int(len(mothers_df)),
    }
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    (registry_out.parent / "timeshift_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote shifted registry: {registry_out}")
    print(f"Wrote shifted CTG: {ctg_out}")
    print(f"Wrote shift key (keep with intermediate data): {key_out}")
    print(f"Wrote BabyID -> MotherID table: {mothers_out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8: time-shift the final outputs.")
    parser.add_argument("--stage3", default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--all-sessions-in", default=DEFAULT_STAGE3_ALL_SESSIONS_DIR)
    parser.add_argument("--registry-in", default=DEFAULT_STAGE7_REGISTRY_CSV)
    parser.add_argument("--ctg-in", default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument("--registry-out", default=DEFAULT_STAGE8_REGISTRY_CSV)
    parser.add_argument("--ctg-out", default=DEFAULT_STAGE8_CTG_PARQUET)
    parser.add_argument("--all-sessions-out", default=DEFAULT_STAGE8_ALL_SESSIONS_DIR)
    parser.add_argument("--key-out", default=DEFAULT_STAGE8_KEY_FILE)
    parser.add_argument("--mothers-out", default=DEFAULT_STAGE8_MOTHERS_CSV)
    parser.add_argument("--max-days", type=int, default=DEFAULT_TIMESHIFT_MAX_DAYS)
    parser.add_argument("--jitter-min", type=float, default=DEFAULT_TIMESHIFT_JITTER_MIN)
    parser.add_argument("--jitter-max", type=float, default=DEFAULT_TIMESHIFT_JITTER_MAX)
    parser.add_argument(
        "--no-all-sessions", action="store_true", help="Skip the pretraining export."
    )
    args = parser.parse_args()
    try:
        time_shift_outputs(
            stage3_path=args.stage3,
            all_sessions_in=None if args.no_all_sessions else args.all_sessions_in,
            registry_in=args.registry_in,
            ctg_in=args.ctg_in,
            registry_out=args.registry_out,
            ctg_out=args.ctg_out,
            all_sessions_out=args.all_sessions_out,
            key_out=args.key_out,
            mothers_out=args.mothers_out,
            max_days=args.max_days,
            jitter_min=args.jitter_min,
            jitter_max=args.jitter_max,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
