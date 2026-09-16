"""Stage 8b: a longer CTG history for the matched pregnancies, without re-running the pipeline.

Stage 7's ``ctg_final.parquet`` holds only the final window (60 min). The stage 3 all-sessions
export holds every session of every pregnancy, so the last N minutes before the end of the
final window can be cut from it for the same BabyIDs, time-shifted with the stage 8 key and
written next to the deliverable as ``ctg_final_<N>m.parquet`` with the same columns as
``ctg_final.parquet``. The cohort and the window end are unchanged; only the look-back grows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from config import (
    DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    DEFAULT_STAGE7_CTG_PARQUET,
    DEFAULT_STAGE8_DIR,
    DEFAULT_STAGE8_KEY_FILE,
)

OUTPUT_COLUMNS = [
    "BabyID",
    "Timestamp",
    "FHR",
    "fhr_stv",
    "toco",
    "Hr1_SignalQuality",
    "Hr1Mode",
    "TocoMode",
]


def _safe(path: str | Path) -> str:
    return str(path).replace("'", "''")


def build_extended_window(
    minutes: int,
    *,
    all_sessions_dir: str | Path = DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    ctg_final: str | Path = DEFAULT_STAGE7_CTG_PARQUET,
    key_file: str | Path = DEFAULT_STAGE8_KEY_FILE,
    out: str | Path | None = None,
) -> dict[str, int]:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    out_path = Path(out) if out else Path(DEFAULT_STAGE8_DIR) / f"ctg_final_{minutes}m.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        f"""
        CREATE TEMP TABLE ends AS
        SELECT BabyID, MAX(Timestamp) AS window_end FROM read_parquet('{_safe(ctg_final)}') GROUP BY BabyID
        """
    )
    available = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_safe(all_sessions_dir)}/*.parquet', union_by_name=true)"
        ).fetchall()
    }
    cols = [c for c in OUTPUT_COLUMNS if c in available]
    select = ", ".join(
        "CAST(a.Timestamp + to_days(k.shift_days) AS TIMESTAMP) AS Timestamp"
        if c == "Timestamp"
        else f"a.{c}"
        for c in cols
    )
    con.execute(
        f"""
        COPY (
            SELECT {select}
            FROM read_parquet('{_safe(all_sessions_dir)}/*.parquet', union_by_name=true) a
            JOIN ends e USING (BabyID)
            JOIN read_parquet('{_safe(key_file)}') k USING (BabyID)
            WHERE a.Timestamp > e.window_end - INTERVAL {int(minutes)} MINUTE
              AND a.Timestamp <= e.window_end
        ) TO '{_safe(out_path)}' (FORMAT PARQUET)
        """
    )
    rows = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT BabyID) FROM read_parquet('{_safe(out_path)}')"
    ).fetchone()
    n_rows, n_babies = (int(rows[0]), int(rows[1])) if rows else (0, 0)
    matched_row = con.execute("SELECT COUNT(*) FROM ends").fetchone()
    n_matched = int(matched_row[0]) if matched_row else 0
    summary = {"minutes": minutes, "rows": n_rows, "babies": n_babies, "matched_babies": n_matched}
    print(summary)
    print(f"Wrote {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut the last N minutes before the final window end from all sessions."
    )
    parser.add_argument("--minutes", type=int, default=180)
    parser.add_argument("--all-sessions", default=DEFAULT_STAGE3_ALL_SESSIONS_DIR)
    parser.add_argument("--ctg-final", default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument("--key", default=DEFAULT_STAGE8_KEY_FILE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    build_extended_window(
        args.minutes,
        all_sessions_dir=args.all_sessions,
        ctg_final=args.ctg_final,
        key_file=args.key,
        out=args.out,
    )


if __name__ == "__main__":
    main()
