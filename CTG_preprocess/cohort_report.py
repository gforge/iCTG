from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from config import (
    DEFAULT_PATIENT_CSV,
    DEFAULT_STAGE0_DIR,
    DEFAULT_STAGE1_DIR,
    DEFAULT_STAGE2_DIR,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE3_GAP_MINUTES,
    DEFAULT_STAGE3_PREG_GAP_DAYS,
    DEFAULT_STAGE4_DIR,
    DEFAULT_STAGE5_5_OUTPUT_FILE,
    DEFAULT_STAGE5_OUTPUT_FILE,
    DEFAULT_STAGE6_DIR,
    DEFAULT_STAGE7_CTG_PARQUET,
    DEFAULT_STAGE7_REGISTRY_CSV,
)


def _safe(path: str) -> str:
    return path.replace("'", "''")


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _source_sql(path: str | Iterable[str]) -> tuple[str | None, int, int]:
    if isinstance(path, (list, tuple)):
        files = [str(p) for p in path if Path(p).exists()]
        if not files:
            return None, 0, 0
        source = "read_parquet([" + ",".join(f"'{_safe(p)}'" for p in files) + "])"
        size = sum(Path(p).stat().st_size for p in files)
        return source, len(files), size

    p = Path(path)
    if p.is_dir():
        files = list(p.rglob("*.parquet"))
        if not files:
            return None, 0, 0
        return (
            f"read_parquet('{_safe(str(p / '**' / '*.parquet'))}')",
            len(files),
            sum(f.stat().st_size for f in files),
        )
    if p.exists():
        return f"read_parquet('{_safe(str(p))}')", 1, p.stat().st_size
    return None, 0, 0


def _metadata_row_count(path: str | Iterable[str]) -> int | None:
    if isinstance(path, (list, tuple)):
        files = [Path(p) for p in path if Path(p).exists()]
    else:
        p = Path(path)
        if p.is_dir():
            files = list(p.rglob("*.parquet"))
        elif p.exists() and p.suffix == ".parquet":
            files = [p]
        else:
            files = []
    if not files:
        return None
    total = 0
    for file in files:
        parquet_file = pq.ParquetFile(file)
        try:
            total += parquet_file.metadata.num_rows
        finally:
            parquet_file.close()
    return total


def _columns(con: duckdb.DuckDBPyConnection, source_sql: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()]


def _basic_counts(
    con: duckdb.DuckDBPyConnection, source_sql: str, cols: list[str]
) -> tuple[int, int | None, int | None]:
    select_parts = ["COUNT(*) AS rows"]
    if "BabyID" in cols:
        select_parts.append("COUNT(DISTINCT BabyID) AS babies")
    if "PatientID" in cols:
        select_parts.append("COUNT(DISTINCT PatientID) AS patients")
    row = con.execute(f"SELECT {', '.join(select_parts)} FROM {source_sql}").fetchone()
    rows = int(row[0] or 0)
    idx = 1
    babies = int(row[idx] or 0) if "BabyID" in cols else None
    if "BabyID" in cols:
        idx += 1
    patients = int(row[idx] or 0) if "PatientID" in cols else None
    return rows, babies, patients


def _pregnancy_count_pre_stage3(
    con: duckdb.DuckDBPyConnection,
    source_sql: str,
    gap_minutes: int,
    preg_gap_days: int,
) -> int:
    row = con.execute(
        f"""
        WITH ordered AS (
            SELECT
                PatientID,
                Timestamp,
                Timestamp - LAG(Timestamp) OVER (PARTITION BY PatientID ORDER BY Timestamp) AS gap
            FROM {source_sql}
        ),
        sessioned AS (
            SELECT
                *,
                SUM(
                    CASE WHEN gap IS NULL OR gap > INTERVAL '{gap_minutes} minutes' THEN 1 ELSE 0 END
                ) OVER (PARTITION BY PatientID ORDER BY Timestamp) AS session_id
            FROM ordered
        ),
        session_end AS (
            SELECT PatientID, session_id, MAX(Timestamp) AS session_end
            FROM sessioned
            GROUP BY PatientID, session_id
        ),
        preg_sessions AS (
            SELECT
                PatientID,
                SUM(
                    CASE
                        WHEN prev_end IS NULL OR session_end - prev_end > INTERVAL '{preg_gap_days} days' THEN 1
                        ELSE 0
                    END
                ) OVER (PARTITION BY PatientID ORDER BY session_end) AS pregnancy_id
            FROM (
                SELECT
                    *,
                    LAG(session_end) OVER (PARTITION BY PatientID ORDER BY session_end) AS prev_end
                FROM session_end
            )
        )
        SELECT COUNT(*)
        FROM (SELECT DISTINCT PatientID, pregnancy_id FROM preg_sessions)
        """
    ).fetchone()
    return int(row[0] or 0)


def _pre_stage3_date_cluster_stats(
    con: duckdb.DuckDBPyConnection,
    source_sql: str,
    table_name: str,
    preg_gap_days: int,
) -> tuple[int, int, int]:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table_name}_dates AS
        SELECT PatientID, CAST(Timestamp AS DATE) AS ctg_day
        FROM {source_sql}
        WHERE PatientID IS NOT NULL AND Timestamp IS NOT NULL
        GROUP BY PatientID, CAST(Timestamp AS DATE)
        """
    )
    patients = con.execute(f"SELECT COUNT(DISTINCT PatientID) FROM {table_name}_dates").fetchone()[
        0
    ]
    overlap = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT d.PatientID
            FROM {table_name}_dates d
            JOIN registry_ids r USING (PatientID)
        )
        """
    ).fetchone()[0]
    pregnancies = con.execute(
        f"""
        WITH dated AS (
            SELECT
                PatientID,
                ctg_day,
                LAG(ctg_day) OVER (PARTITION BY PatientID ORDER BY ctg_day) AS prev_day
            FROM {table_name}_dates
        ),
        pregnancy_starts AS (
            SELECT
                PatientID,
                CASE
                    WHEN prev_day IS NULL
                      OR date_diff('day', prev_day, ctg_day) > {preg_gap_days}
                    THEN 1
                    ELSE 0
                END AS is_start
            FROM dated
        )
        SELECT SUM(is_start)
        FROM pregnancy_starts
        """
    ).fetchone()[0]
    return int(patients or 0), int(pregnancies or 0), int(overlap or 0)


def _registry_overlap(con: duckdb.DuckDBPyConnection, source_sql: str) -> int:
    row = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT s.PatientID
            FROM {source_sql} s
            JOIN registry_ids r USING (PatientID)
        )
        """
    ).fetchone()
    return int(row[0] or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact cohort reduction report across stages.")
    parser.add_argument("--registry-csv", type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--gap-minutes", type=int, default=DEFAULT_STAGE3_GAP_MINUTES)
    parser.add_argument("--preg-gap-days", type=int, default=DEFAULT_STAGE3_PREG_GAP_DAYS)
    parser.add_argument(
        "--include-pre-stage3-pregnancies",
        action="store_true",
        help=(
            "Also estimate pregnancy counts for raw/stage1/stage2. "
            "Defaults to a low-memory PatientID/date clustering algorithm."
        ),
    )
    parser.add_argument(
        "--pre-stage3-pregnancy-method",
        choices=["date-clusters", "exact-windows"],
        default="date-clusters",
        help=(
            "Method for --include-pre-stage3-pregnancies. date-clusters is much lower memory; "
            "exact-windows reproduces the second-level session logic but can need very large RAM."
        ),
    )
    parser.add_argument(
        "--exact-pre-stage3-counts",
        action="store_true",
        help="Scan raw/stage1/stage2 to count distinct patients and registry overlap instead of using only parquet metadata row counts.",
    )
    args = parser.parse_args()

    start_all = time.perf_counter()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    safe_registry = _safe(args.registry_csv)
    print("Building registry PatientID lookup...")
    con.execute(
        f"""
        CREATE TEMP TABLE registry_ids AS
        SELECT DISTINCT substr(reg_digits, 1, 8) || '-' || substr(reg_digits, 9, 4) AS PatientID
        FROM (
            SELECT regexp_replace(CAST(personnummer_mor AS VARCHAR), '[^0-9]', '', 'g') AS reg_digits
            FROM read_csv_auto('{safe_registry}', delim=';', header=true)
            WHERE personnummer_mor IS NOT NULL
        )
        WHERE reg_digits IS NOT NULL AND length(reg_digits) >= 12
        """
    )
    reg_patients = con.execute("SELECT COUNT(*) FROM registry_ids").fetchone()[0]
    print(f"registry_person_ids: {reg_patients}")

    stages = [
        ("raw", DEFAULT_STAGE0_DIR),
        ("stage1", DEFAULT_STAGE1_DIR),
        ("stage2", DEFAULT_STAGE2_DIR),
        ("stage3", DEFAULT_STAGE3_DIR),
        ("stage4", DEFAULT_STAGE4_DIR),
        ("stage5", DEFAULT_STAGE5_OUTPUT_FILE),
        ("stage5_5", DEFAULT_STAGE5_5_OUTPUT_FILE),
        ("stage6", DEFAULT_STAGE6_DIR),
        ("stage7_ctg", DEFAULT_STAGE7_CTG_PARQUET),
        ("stage7_registry", DEFAULT_STAGE7_REGISTRY_CSV),
    ]

    print(
        "stage,rows,patients,babies,pregnancies,registry_overlap_patients,files,size_mb,row_retention_from_previous_pct,baby_retention_from_previous_pct"
    )
    completed = 0
    total = len(stages)
    prev_rows: int | None = None
    prev_babies: int | None = None
    for name, path in stages:
        stage_start = time.perf_counter()
        print(f"Analyzing {name} ({completed + 1}/{total})...")
        source_sql, file_count, byte_size = _source_sql(path)
        if not source_sql:
            print(f"{name},missing,-,-,-,-,0,0.0,-,-")
            completed += 1
            continue

        is_csv = name == "stage7_registry"
        is_pre_stage3 = name in {"raw", "stage1", "stage2"}
        if is_pre_stage3 and not (
            args.exact_pre_stage3_counts or args.include_pre_stage3_pregnancies
        ):
            metadata_rows = _metadata_row_count(path)
            if metadata_rows is None:
                print(f"{name},missing,-,-,-,-,0,0.0,-,-")
                completed += 1
                continue
            rows = metadata_rows
            babies = None
            patients = None
            pregnancies = None
            overlap = None
            size_mb = byte_size / (1024 * 1024)
            row_retention = rows / prev_rows * 100.0 if prev_rows else None
            row_retention_out = f"{row_retention:.2f}" if row_retention is not None else "-"
            print(f"{name},{rows},-,-,-,-,{file_count},{size_mb:.1f},{row_retention_out},-")
            completed += 1
            prev_rows = rows
            elapsed_stage = time.perf_counter() - stage_start
            elapsed_all = time.perf_counter() - start_all
            avg_stage = elapsed_all / completed if completed else 0.0
            remaining = avg_stage * (total - completed)
            print(
                f"  done {name} in {_fmt_seconds(elapsed_stage)}; "
                f"elapsed {_fmt_seconds(elapsed_all)}; "
                f"eta {_fmt_seconds(remaining)}"
            )
            continue

        if (
            is_pre_stage3
            and args.include_pre_stage3_pregnancies
            and args.pre_stage3_pregnancy_method == "date-clusters"
        ):
            metadata_rows = _metadata_row_count(path)
            if metadata_rows is None:
                print(f"{name},missing,-,-,-,-,0,0.0,-,-")
                completed += 1
                continue
            print(
                f"  {name}: counting pregnancies from distinct PatientID/date clusters "
                f"(gap > {args.preg_gap_days} days)..."
            )
            patients, pregnancies, overlap = _pre_stage3_date_cluster_stats(
                con,
                source_sql,
                table_name=f"{name}_pre_stage3",
                preg_gap_days=args.preg_gap_days,
            )
            rows = metadata_rows
            babies = None
            size_mb = byte_size / (1024 * 1024)
            row_retention = rows / prev_rows * 100.0 if prev_rows else None
            row_retention_out = f"{row_retention:.2f}" if row_retention is not None else "-"
            print(
                f"{name},{rows},{patients},-,{pregnancies},{overlap},"
                f"{file_count},{size_mb:.1f},{row_retention_out},-"
            )
            completed += 1
            prev_rows = rows
            elapsed_stage = time.perf_counter() - stage_start
            elapsed_all = time.perf_counter() - start_all
            avg_stage = elapsed_all / completed if completed else 0.0
            remaining = avg_stage * (total - completed)
            print(
                f"  done {name} in {_fmt_seconds(elapsed_stage)}; "
                f"elapsed {_fmt_seconds(elapsed_all)}; "
                f"eta {_fmt_seconds(remaining)}"
            )
            continue

        if is_csv:
            safe_path = _safe(str(path))
            source_sql = f"read_csv_auto('{safe_path}', header=true)"
        cols = _columns(con, source_sql)
        rows, babies, patients = _basic_counts(con, source_sql, cols)
        if "BabyID" in cols:
            pregnancies = babies
        elif args.include_pre_stage3_pregnancies and "PatientID" in cols and "Timestamp" in cols:
            print(f"  {name}: counting pregnancies from PatientID/Timestamp sessions...")
            pregnancies = _pregnancy_count_pre_stage3(
                con,
                source_sql,
                gap_minutes=args.gap_minutes,
                preg_gap_days=args.preg_gap_days,
            )
        else:
            pregnancies = None

        overlap = _registry_overlap(con, source_sql) if "PatientID" in cols and not is_csv else None
        size_mb = byte_size / (1024 * 1024)
        row_retention = rows / prev_rows * 100.0 if prev_rows else None
        baby_retention = (
            babies / prev_babies * 100.0 if babies is not None and prev_babies else None
        )
        row_retention_out = f"{row_retention:.2f}" if row_retention is not None else "-"
        baby_retention_out = f"{baby_retention:.2f}" if baby_retention is not None else "-"
        print(
            f"{name},{rows},{patients if patients is not None else '-'},"
            f"{babies if babies is not None else '-'},"
            f"{pregnancies if pregnancies is not None else '-'},"
            f"{overlap if overlap is not None else '-'},"
            f"{file_count},{size_mb:.1f},{row_retention_out},{baby_retention_out}"
        )
        completed += 1
        prev_rows = rows
        if babies is not None:
            prev_babies = babies
        elapsed_stage = time.perf_counter() - stage_start
        elapsed_all = time.perf_counter() - start_all
        avg_stage = elapsed_all / completed if completed else 0.0
        remaining = avg_stage * (total - completed)
        print(
            f"  done {name} in {_fmt_seconds(elapsed_stage)}; "
            f"elapsed {_fmt_seconds(elapsed_all)}; "
            f"eta {_fmt_seconds(remaining)}"
        )

    print(f"Total runtime: {_fmt_seconds(time.perf_counter() - start_all)}")


if __name__ == "__main__":
    main()
