from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from config import (
    DEFAULT_PATIENT_CSV,
    DEFAULT_STAGE0_DIR,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE3_PREG_GAP_DAYS,
)


def _safe(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _parquet_sql(path: str | Path, recursive: bool = False) -> str:
    path = Path(path)
    if path.is_dir():
        pattern = path / ("**/*.parquet" if recursive else "*.parquet")
    else:
        pattern = path
    return f"read_parquet('{_safe(pattern)}')"


def _build_registry(con: duckdb.DuckDBPyConnection, registry_csv: str | Path) -> None:
    safe_registry = _safe(registry_csv)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE registry_births AS
        SELECT
            row_number() OVER () AS reg_row,
            substr(reg_digits, 1, 8) || '-' || substr(reg_digits, 9, 4) AS PatientID,
            TRY_CAST(NULLIF(trim(CAST(forlossningsdatum_fv1 AS VARCHAR)), '') AS DATE) AS birth_day
        FROM (
            SELECT
                regexp_replace(CAST(personnummer_mor AS VARCHAR), '[^0-9]', '', 'g') AS reg_digits,
                forlossningsdatum_fv1
            FROM read_csv_auto('{safe_registry}', delim=';', header=true)
            WHERE personnummer_mor IS NOT NULL
        )
        WHERE reg_digits IS NOT NULL
          AND length(reg_digits) >= 12
          AND birth_day IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE registry_pregnancies AS
        SELECT
            PatientID,
            birth_day,
            COUNT(*) AS registry_birth_rows
        FROM registry_births
        GROUP BY PatientID, birth_day
        """
    )


def _build_ctg_from_stage3(con: duckdb.DuckDBPyConnection, stage3_path: str | Path) -> None:
    source = _parquet_sql(stage3_path)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ctg_pregnancies AS
        SELECT
            BabyID AS ctg_episode_id,
            MIN(PatientID) AS PatientID,
            CAST(MAX(Timestamp) AS DATE) AS ctg_date,
            MIN(Timestamp) AS window_start,
            MAX(Timestamp) AS window_end,
            COUNT(*) AS rows
        FROM {source}
        GROUP BY BabyID
        """
    )


def _build_ctg_from_raw_dates(
    con: duckdb.DuckDBPyConnection,
    raw_path: str | Path,
    preg_gap_days: int,
) -> None:
    source = _parquet_sql(raw_path)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ctg_dates AS
        SELECT PatientID, CAST(Timestamp AS DATE) AS ctg_day
        FROM {source}
        WHERE PatientID IS NOT NULL AND Timestamp IS NOT NULL
        GROUP BY PatientID, CAST(Timestamp AS DATE)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ctg_pregnancies AS
        WITH dated AS (
            SELECT
                PatientID,
                ctg_day,
                CASE
                    WHEN lag(ctg_day) OVER (PARTITION BY PatientID ORDER BY ctg_day) IS NULL
                      OR date_diff(
                            'day',
                            lag(ctg_day) OVER (PARTITION BY PatientID ORDER BY ctg_day),
                            ctg_day
                         ) > {preg_gap_days}
                    THEN 1
                    ELSE 0
                END AS new_pregnancy
            FROM ctg_dates
        ),
        grouped AS (
            SELECT
                PatientID,
                ctg_day,
                SUM(new_pregnancy) OVER (PARTITION BY PatientID ORDER BY ctg_day) AS pregnancy_id
            FROM dated
        )
        SELECT
            PatientID || '|' || CAST(pregnancy_id AS VARCHAR) AS ctg_episode_id,
            PatientID,
            MAX(ctg_day) AS ctg_date,
            MIN(ctg_day) AS window_start,
            MAX(ctg_day) AS window_end,
            COUNT(*) AS recording_days
        FROM grouped
        GROUP BY PatientID, pregnancy_id
        """
    )


def _build_matches(con: duckdb.DuckDBPyConnection, include_birth_day_after_ctg: bool) -> None:
    if include_birth_day_after_ctg:
        day_predicate = "r.birth_day BETWEEN c.ctg_date AND c.ctg_date + INTERVAL 1 DAY"
    else:
        # Same rule as Stage 7: CTG date is the birth date or the day before birth.
        day_predicate = "c.ctg_date = r.birth_day OR c.ctg_date = r.birth_day - INTERVAL 1 DAY"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE match_candidates AS
        SELECT
            r.reg_row,
            r.PatientID,
            r.birth_day,
            c.ctg_episode_id,
            c.ctg_date
        FROM registry_births r
        JOIN ctg_pregnancies c
          ON r.PatientID = c.PatientID
         AND ({day_predicate})
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pregnancy_match_candidates AS
        SELECT
            r.PatientID,
            r.birth_day,
            c.ctg_episode_id,
            c.ctg_date
        FROM registry_pregnancies r
        JOIN ctg_pregnancies c
          ON r.PatientID = c.PatientID
         AND (c.ctg_date = r.birth_day OR c.ctg_date = r.birth_day - INTERVAL 1 DAY)
        """
    )


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no rows: {sql}")
    return int(row[0] or 0)


def _print_summary(con: duckdb.DuckDBPyConnection, cutoff_date: str) -> None:
    print("\nSummary")
    rows = [
        ("registry_birth_rows", "SELECT COUNT(*) FROM registry_births"),
        ("registry_pregnancy_episodes", "SELECT COUNT(*) FROM registry_pregnancies"),
        ("ctg_pregnancy_episodes", "SELECT COUNT(*) FROM ctg_pregnancies"),
        (
            f"ctg_pregnancy_episodes_before_{cutoff_date}",
            f"SELECT COUNT(*) FROM ctg_pregnancies WHERE ctg_date < DATE '{cutoff_date}'",
        ),
        (
            f"ctg_pregnancy_episodes_from_{cutoff_date}",
            f"SELECT COUNT(*) FROM ctg_pregnancies WHERE ctg_date >= DATE '{cutoff_date}'",
        ),
        (
            "registry_birth_rows_with_ctg_match",
            "SELECT COUNT(DISTINCT reg_row) FROM match_candidates",
        ),
        (
            "registry_pregnancy_episodes_with_ctg_match",
            "SELECT COUNT(*) FROM (SELECT DISTINCT PatientID, birth_day FROM pregnancy_match_candidates)",
        ),
        (
            "ctg_pregnancy_episodes_with_registry_match",
            "SELECT COUNT(DISTINCT ctg_episode_id) FROM match_candidates",
        ),
        (
            "ctg_pregnancy_episodes_without_registry_match",
            """
            SELECT COUNT(*)
            FROM ctg_pregnancies c
            WHERE NOT EXISTS (
                SELECT 1 FROM match_candidates m
                WHERE m.ctg_episode_id = c.ctg_episode_id
            )
            """,
        ),
        (
            f"ctg_pregnancy_episodes_without_registry_match_from_{cutoff_date}",
            f"""
            SELECT COUNT(*)
            FROM ctg_pregnancies c
            WHERE c.ctg_date >= DATE '{cutoff_date}'
              AND NOT EXISTS (
                  SELECT 1 FROM match_candidates m
                  WHERE m.ctg_episode_id = c.ctg_episode_id
              )
            """,
        ),
        (
            "registry_birth_rows_with_multiple_ctg_matches",
            """
            SELECT COUNT(*)
            FROM (
                SELECT reg_row
                FROM match_candidates
                GROUP BY reg_row
                HAVING COUNT(DISTINCT ctg_episode_id) > 1
            )
            """,
        ),
        (
            "ctg_pregnancy_episodes_matching_multiple_registry_birth_rows",
            """
            SELECT COUNT(*)
            FROM (
                SELECT ctg_episode_id
                FROM match_candidates
                GROUP BY ctg_episode_id
                HAVING COUNT(DISTINCT reg_row) > 1
            )
            """,
        ),
    ]
    for label, sql in rows:
        print(f"{label},{_scalar(con, sql)}")


def _print_year_tables(con: duckdb.DuckDBPyConnection, cutoff_date: str) -> None:
    print("\nRegistry birth rows by birth year")
    print("year,registry_birth_rows,matched_birth_rows,match_pct")
    for year, total, matched in con.execute(
        """
        WITH per_year AS (
            SELECT EXTRACT(YEAR FROM birth_day)::INTEGER AS year, COUNT(*) AS total
            FROM registry_births
            GROUP BY year
        ),
        matched AS (
            SELECT EXTRACT(YEAR FROM birth_day)::INTEGER AS year, COUNT(DISTINCT reg_row) AS matched
            FROM match_candidates
            GROUP BY year
        )
        SELECT p.year, p.total, COALESCE(m.matched, 0)
        FROM per_year p
        LEFT JOIN matched m USING (year)
        ORDER BY p.year
        """
    ).fetchall():
        pct = matched / total * 100.0 if total else 0.0
        print(f"{year},{total},{matched},{pct:.2f}")

    print("\nCTG pregnancy episodes by CTG year")
    print("year,ctg_pregnancies,matched_to_registry,unmatched,unmatched_pct")
    for year, total, matched in con.execute(
        """
        WITH per_year AS (
            SELECT EXTRACT(YEAR FROM ctg_date)::INTEGER AS year, COUNT(*) AS total
            FROM ctg_pregnancies
            GROUP BY year
        ),
        matched AS (
            SELECT EXTRACT(YEAR FROM c.ctg_date)::INTEGER AS year, COUNT(DISTINCT c.ctg_episode_id) AS matched
            FROM ctg_pregnancies c
            JOIN match_candidates m USING (ctg_episode_id)
            GROUP BY year
        )
        SELECT p.year, p.total, COALESCE(m.matched, 0)
        FROM per_year p
        LEFT JOIN matched m USING (year)
        ORDER BY p.year
        """
    ).fetchall():
        unmatched = total - matched
        pct = unmatched / total * 100.0 if total else 0.0
        print(f"{year},{total},{matched},{unmatched},{pct:.2f}")

    print(f"\nFrom {cutoff_date} onward")
    reg_total = _scalar(
        con, f"SELECT COUNT(*) FROM registry_births WHERE birth_day >= DATE '{cutoff_date}'"
    )
    reg_matched = _scalar(
        con,
        f"SELECT COUNT(DISTINCT reg_row) FROM match_candidates WHERE birth_day >= DATE '{cutoff_date}'",
    )
    ctg_total = _scalar(
        con, f"SELECT COUNT(*) FROM ctg_pregnancies WHERE ctg_date >= DATE '{cutoff_date}'"
    )
    ctg_matched = _scalar(
        con,
        f"""
        SELECT COUNT(DISTINCT c.ctg_episode_id)
        FROM ctg_pregnancies c
        JOIN match_candidates m USING (ctg_episode_id)
        WHERE c.ctg_date >= DATE '{cutoff_date}'
        """,
    )
    print("metric,total,matched,unmatched,matched_pct")
    print(
        f"registry_birth_rows,{reg_total},{reg_matched},{reg_total - reg_matched},"
        f"{(reg_matched / reg_total * 100.0 if reg_total else 0.0):.2f}"
    )
    print(
        f"ctg_pregnancy_episodes,{ctg_total},{ctg_matched},{ctg_total - ctg_matched},"
        f"{(ctg_matched / ctg_total * 100.0 if ctg_total else 0.0):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pregnancy-level overlap between CTG and gravniva."
    )
    parser.add_argument("--registry-csv", type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--stage3", type=str, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--raw", type=str, default=DEFAULT_STAGE0_DIR)
    parser.add_argument(
        "--ctg-source",
        choices=["stage3", "raw-dates"],
        default="stage3",
        help=(
            "stage3 is fast and exact for post-stage3 CTG pregnancy episodes. "
            "raw-dates scans raw CTG and approximates pregnancies from recording dates."
        ),
    )
    parser.add_argument("--preg-gap-days", type=int, default=DEFAULT_STAGE3_PREG_GAP_DAYS)
    parser.add_argument("--cutoff-date", type=str, default="2015-01-01")
    args = parser.parse_args()

    start = time.perf_counter()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass

    print("Building registry birth tables...")
    _build_registry(con, args.registry_csv)
    if args.ctg_source == "stage3":
        print("Building CTG pregnancy table from Stage 3 BabyIDs...")
        _build_ctg_from_stage3(con, args.stage3)
    else:
        print("Building approximate CTG pregnancy table from raw PatientID/date pairs...")
        _build_ctg_from_raw_dates(con, args.raw, args.preg_gap_days)

    print("Matching by PatientID and CTG date equal to birth date or the day before...")
    _build_matches(con, include_birth_day_after_ctg=False)
    _print_summary(con, args.cutoff_date)
    _print_year_tables(con, args.cutoff_date)
    print(f"\nRuntime,{_fmt_seconds(time.perf_counter() - start)}")


if __name__ == "__main__":
    main()
