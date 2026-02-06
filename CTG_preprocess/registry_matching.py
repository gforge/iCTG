from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from config import (
    DEFAULT_PATIENT_CSV,
    DEFAULT_STAGE5_5_OUTPUT_FILE,
    DEFAULT_STAGE6_DIR,
    DEFAULT_STAGE7_CTG_PARQUET,
    DEFAULT_STAGE7_REGISTRY_CSV,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def registry_match(
    registry_csv: str | Path,
    stage5_5_file: str | Path,
    stage6_dir: str | Path,
    registry_out: str | Path,
    ctg_out: str | Path,
    show_progress: bool = True,
) -> None:
    registry_csv = Path(registry_csv)
    stage5_5_file = Path(stage5_5_file)
    stage6_dir = Path(stage6_dir)
    registry_out = Path(registry_out)
    ctg_out = Path(ctg_out)

    _ensure_parent(registry_out)
    _ensure_parent(ctg_out)

    con = duckdb.connect()
    if show_progress:
        try:
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA progress_bar_time=5")
        except Exception:
            pass
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass

    safe_registry = str(registry_csv).replace("'", "''")
    safe_stage5_5 = str(stage5_5_file).replace("'", "''")
    safe_stage6 = str(stage6_dir).replace("'", "''")

    con.execute(
        f"""
        CREATE VIEW reg_raw AS
        SELECT * FROM read_csv_auto('{safe_registry}', delim=';', header=true)
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE reg AS
        SELECT
            row_number() OVER () AS reg_row,
            regexp_replace(CAST(personnummer_mor AS VARCHAR), '[^0-9]', '', 'g') AS reg_digits,
            TRY_CAST(apgar_5_min AS INTEGER) AS apgar5,
            TRY_CAST(forlossningsdatum_fv1 AS DATE) AS birth_day
        FROM reg_raw
        WHERE personnummer_mor IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE reg_clean AS
        SELECT
            reg_row,
            birth_day,
            apgar5,
            substr(reg_digits, 1, 8) || '-' || substr(reg_digits, 9, 4) AS PatientID
        FROM reg
        WHERE reg_digits IS NOT NULL
          AND length(reg_digits) >= 12
          AND apgar5 IS NOT NULL
          AND birth_day IS NOT NULL
        """
    )

    con.execute(
        f"""
        CREATE VIEW s55 AS
        SELECT BabyID, PatientID, ctg_date
        FROM read_parquet('{safe_stage5_5}')
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE map AS
        SELECT DISTINCT BabyID, PatientID, ctg_date
        FROM s55
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE matches AS
        SELECT
            r.reg_row,
            r.PatientID,
            r.birth_day,
            r.apgar5,
            m.BabyID,
            m.ctg_date
        FROM reg_clean r
        JOIN map m
          ON r.PatientID = m.PatientID
         AND (m.ctg_date = r.birth_day OR m.ctg_date = r.birth_day - INTERVAL 1 DAY)
        """
    )

    multi_rows = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT reg_row, COUNT(*) AS cnt
            FROM matches
            GROUP BY reg_row
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    con.execute(
        """
        CREATE TEMP TABLE unique_matches AS
        WITH counted AS (
            SELECT reg_row, COUNT(*) AS cnt
            FROM matches
            GROUP BY reg_row
        )
        SELECT m.*
        FROM matches m
        JOIN counted c USING (reg_row)
        WHERE c.cnt = 1
        """
    )

    total_rows = con.execute("SELECT COUNT(*) FROM reg_raw").fetchone()[0]
    clean_rows = con.execute("SELECT COUNT(*) FROM reg_clean").fetchone()[0]
    match_rows = con.execute("SELECT COUNT(*) FROM unique_matches").fetchone()[0]

    print(f"Registry rows total: {total_rows}")
    print(f"Registry rows with valid apgar/birth_day: {clean_rows}")
    print(f"Matched rows: {match_rows}")
    if multi_rows:
        print(f"WARNING: {multi_rows} registry rows matched multiple BabyIDs and were dropped.")

    con.execute(
        f"""
        COPY (
            SELECT BabyID, birth_day, apgar5
            FROM unique_matches
            ORDER BY BabyID
        ) TO '{str(registry_out).replace("'", "''")}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE matched_babies AS
        SELECT DISTINCT BabyID FROM unique_matches
        """
    )

    con.execute(
        f"""
        CREATE VIEW s6 AS
        SELECT * FROM read_parquet('{safe_stage6}/**/*.parquet')
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT s6.BabyID, s6.Timestamp, s6.FHR, s6.toco
            FROM s6
            JOIN matched_babies mb USING (BabyID)
        ) TO '{str(ctg_out).replace("'", "''")}'
        (FORMAT PARQUET)
        """
    )

    print(f"Wrote registry CSV: {registry_out}")
    print(f"Wrote CTG parquet: {ctg_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 registry matching and anonymized output.")
    parser.add_argument("--registry-csv", type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--stage5-5", type=str, default=DEFAULT_STAGE5_5_OUTPUT_FILE)
    parser.add_argument("--stage6", type=str, default=DEFAULT_STAGE6_DIR)
    parser.add_argument("--registry-out", type=str, default=DEFAULT_STAGE7_REGISTRY_CSV)
    parser.add_argument("--ctg-out", type=str, default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    registry_match(
        registry_csv=args.registry_csv,
        stage5_5_file=args.stage5_5,
        stage6_dir=args.stage6,
        registry_out=args.registry_out,
        ctg_out=args.ctg_out,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
