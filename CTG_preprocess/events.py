"""Stage 9: link the clinician event export (ExportSignatures, converted by ``ictg-signatures``)
to pregnancies and write an anonymized, time-shifted events table.

Steps:

1. ``registration_map``: RegistrationID -> PatientID, first and last timestamp, derived from
   the raw stage 0 parquet (one slow scan, cached in stage 9's directory).
2. Pregnancy spans: BabyID -> PatientID (stage 3 main output) and the first/last timestamp of
   all its sessions (stage 3 all-sessions export when present, else the main output).
3. A registration belongs to the pregnancy of the same mother whose span, widened by
   ``DEFAULT_EVENT_LINK_MARGIN_HOURS``, contains the registration start; the nearest span
   wins if several qualify. Events inherit the registration's BabyID.
4. ``events_linked.parquet`` (stage 9, restricted): BabyID, event time, typed fields, keyword
   flags from the note text, and the note text itself.
5. ``stage_8_timeshift/events.parquet`` (deliverable): the same without note text, with the
   event time shifted by the stage 8 key so it lines up with the shifted CTG.

Only counts are printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from config import (
    DEFAULT_EVENT_LINK_MARGIN_HOURS,
    DEFAULT_EVENT_NOTE_FLAGS,
    DEFAULT_EVENTS_DIR,
    DEFAULT_STAGE0_DIR,
    DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE8_EVENTS_PARQUET,
    DEFAULT_STAGE8_KEY_FILE,
    DEFAULT_STAGE9_EVENTS,
    DEFAULT_STAGE9_REGISTRATION_MAP,
)

EVENT_VALUE_COLUMNS = [
    ("EventType", "event_type"),
    ("BaseLine", "ctg_baseline"),
    ("Variability", "ctg_variability"),
    ("Acceleration", "ctg_accelerations"),
    ("Decelerations", "ctg_decelerations"),
    ("Stage", "ctg_stage"),
    ("Status", "ctg_status"),
    ("Twin", "twin"),
    ("Hr", "maternal_hr"),
    ("Mspo2", "maternal_spo2"),
    ("HrInvalid", "maternal_hr_invalid"),
    ("Systolic", "bp_systolic"),
    ("Diastolic", "bp_diastolic"),
    ("Mean", "bp_mean"),
    ("NibpHR", "bp_hr"),
    ("Lactate", "scalp_lactate"),
    ("PH", "scalp_ph"),
]


def _safe(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _source(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return f"read_parquet('{_safe(p)}/*.parquet', union_by_name=true)"
    return f"read_parquet('{_safe(p)}')"


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def build_registration_map(
    con: duckdb.DuckDBPyConnection, stage0_dir: str | Path, out: str | Path, refresh: bool = False
) -> int:
    """RegistrationID -> PatientID and time span from stage 0; cached at ``out``."""
    out = Path(out)
    if out.exists() and not refresh:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE regmap AS SELECT * FROM read_parquet('{_safe(out)}')"
        )
        return _count(con, "SELECT COUNT(*) FROM regmap")
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE regmap AS
        SELECT RegistrationID, ANY_VALUE(PatientID) AS PatientID,
               MIN(Timestamp) AS reg_start, MAX(Timestamp) AS reg_end
        FROM {_source(stage0_dir)}
        WHERE RegistrationID IS NOT NULL
        GROUP BY RegistrationID
        """
    )
    con.execute(f"COPY regmap TO '{_safe(out)}' (FORMAT PARQUET)")
    return _count(con, "SELECT COUNT(*) FROM regmap")


def build_pregnancy_spans(
    con: duckdb.DuckDBPyConnection, stage3_dir: str | Path, all_sessions_dir: str | Path | None
) -> int:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE preg AS
        SELECT BabyID, ANY_VALUE(PatientID) AS PatientID,
               MIN(Timestamp) AS span_start, MAX(Timestamp) AS span_end
        FROM {_source(stage3_dir)} GROUP BY BabyID
        """
    )
    if all_sessions_dir is not None and Path(all_sessions_dir).exists():
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE spans AS
            SELECT BabyID, MIN(Timestamp) AS span_start, MAX(Timestamp) AS span_end
            FROM {_source(all_sessions_dir)} GROUP BY BabyID
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE preg AS
            SELECT p.BabyID, p.PatientID,
                   LEAST(p.span_start, COALESCE(s.span_start, p.span_start)) AS span_start,
                   GREATEST(p.span_end, COALESCE(s.span_end, p.span_end)) AS span_end
            FROM preg p LEFT JOIN spans s USING (BabyID)
            """
        )
    return _count(con, "SELECT COUNT(*) FROM preg")


def _note_flag_sql(flags: dict[str, str]) -> str:
    parts = []
    for name, pattern in flags.items():
        safe = pattern.replace("'", "''")
        parts.append(
            f"CASE WHEN e.NoteText IS NULL THEN NULL "
            f"ELSE regexp_matches(lower(e.NoteText), '{safe}') END AS note_{name}"
        )
    return ",\n            ".join(parts)


def link_events(
    con: duckdb.DuckDBPyConnection,
    events_dir: str | Path,
    out: str | Path,
    margin_hours: int = DEFAULT_EVENT_LINK_MARGIN_HOURS,
    note_flags: dict[str, str] | None = None,
) -> dict[str, int]:
    """Assign every event to a BabyID via its registration; write ``events_linked``."""
    flags = DEFAULT_EVENT_NOTE_FLAGS if note_flags is None else note_flags
    con.execute(f"CREATE OR REPLACE TEMP VIEW ev AS SELECT * FROM {_source(events_dir)}")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE reg_baby AS
        SELECT RegistrationID, BabyID
        FROM (
            SELECT r.RegistrationID, p.BabyID,
                   row_number() OVER (
                       PARTITION BY r.RegistrationID
                       ORDER BY abs(date_diff('second', p.span_start, r.reg_start))
                   ) AS rn
            FROM regmap r
            JOIN preg p
              ON p.PatientID = r.PatientID
             AND r.reg_start BETWEEN p.span_start - INTERVAL {int(margin_hours)} HOUR
                                 AND p.span_end + INTERVAL {int(margin_hours)} HOUR
        ) WHERE rn = 1
        """
    )
    value_cols = ",\n            ".join(f"e.{src} AS {dst}" for src, dst in EVENT_VALUE_COLUMNS)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
            rb.BabyID,
            COALESCE(e.Time, e.MedicalTime) AS Timestamp,
            {value_cols},
            (e.NoteText IS NOT NULL) AS note_present,
            length(e.NoteText) AS note_length,
            {_note_flag_sql(flags)},
            e.NoteText
            FROM ev e
            JOIN reg_baby rb USING (RegistrationID)
            WHERE COALESCE(e.Time, e.MedicalTime) IS NOT NULL
        ) TO '{_safe(out)}' (FORMAT PARQUET)
        """
    )
    total = _count(con, "SELECT COUNT(*) FROM ev")
    linked = _count(con, f"SELECT COUNT(*) FROM read_parquet('{_safe(out)}')")
    no_reg = _count(
        con,
        "SELECT COUNT(*) FROM ev e LEFT JOIN regmap r USING (RegistrationID) WHERE r.RegistrationID IS NULL",
    )
    return {
        "events_total": total,
        "events_without_known_registration": no_reg,
        "events_registration_not_in_a_pregnancy": total - no_reg - linked,
        "events_linked": linked,
        "pregnancies_with_events": _count(
            con, f"SELECT COUNT(DISTINCT BabyID) FROM read_parquet('{_safe(out)}')"
        ),
    }


def export_shifted_events(
    con: duckdb.DuckDBPyConnection, linked: str | Path, key_file: str | Path, out: str | Path
) -> int:
    """Deliverable: linked events without the note text, time-shifted with the stage 8 key."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT l.* EXCLUDE (NoteText, Timestamp),
                   CAST(l.Timestamp + to_days(k.shift_days) AS TIMESTAMP) AS Timestamp
            FROM read_parquet('{_safe(linked)}') l
            JOIN read_parquet('{_safe(key_file)}') k USING (BabyID)
        ) TO '{_safe(out)}' (FORMAT PARQUET)
        """
    )
    return _count(con, f"SELECT COUNT(*) FROM read_parquet('{_safe(out)}')")


def run_stage9(
    *,
    events_dir: str | Path = DEFAULT_EVENTS_DIR,
    stage0_dir: str | Path = DEFAULT_STAGE0_DIR,
    stage3_dir: str | Path = DEFAULT_STAGE3_DIR,
    all_sessions_dir: str | Path | None = DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    registration_map: str | Path = DEFAULT_STAGE9_REGISTRATION_MAP,
    linked_out: str | Path = DEFAULT_STAGE9_EVENTS,
    key_file: str | Path = DEFAULT_STAGE8_KEY_FILE,
    shifted_out: str | Path | None = DEFAULT_STAGE8_EVENTS_PARQUET,
    margin_hours: int = DEFAULT_EVENT_LINK_MARGIN_HOURS,
    note_flags: dict[str, str] | None = None,
    refresh_map: bool = False,
) -> dict[str, int]:
    if not Path(events_dir).exists():
        raise FileNotFoundError(
            f"Events parquet not found: {events_dir}. Convert the ExportSignatures files first "
            "(uv run --project <repo> ictg-signatures 'ExportSignatures_*.json' --parquet-out ...)."
        )
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    summary: dict[str, int] = {}
    summary["registrations"] = build_registration_map(
        con, stage0_dir, registration_map, refresh=refresh_map
    )
    summary["pregnancies"] = build_pregnancy_spans(con, stage3_dir, all_sessions_dir)
    summary.update(link_events(con, events_dir, linked_out, margin_hours, note_flags))
    if shifted_out is not None and Path(key_file).exists():
        summary["events_shifted"] = export_shifted_events(con, linked_out, key_file, shifted_out)
    elif shifted_out is not None:
        print(f"Stage 8 key not found ({key_file}); deliverable not written. Run stage 8 first.")
    print(json.dumps(summary, indent=2))
    Path(linked_out).parent.joinpath("events_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 9: link clinician events to pregnancies.")
    parser.add_argument("--events", default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--stage0", default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--stage3", default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--all-sessions", default=DEFAULT_STAGE3_ALL_SESSIONS_DIR)
    parser.add_argument("--registration-map", default=DEFAULT_STAGE9_REGISTRATION_MAP)
    parser.add_argument("--linked-out", default=DEFAULT_STAGE9_EVENTS)
    parser.add_argument("--key", default=DEFAULT_STAGE8_KEY_FILE)
    parser.add_argument("--shifted-out", default=DEFAULT_STAGE8_EVENTS_PARQUET)
    parser.add_argument("--margin-hours", type=int, default=DEFAULT_EVENT_LINK_MARGIN_HOURS)
    parser.add_argument("--refresh-map", action="store_true", help="Rebuild the registration map.")
    args = parser.parse_args()
    run_stage9(
        events_dir=args.events,
        stage0_dir=args.stage0,
        stage3_dir=args.stage3,
        all_sessions_dir=args.all_sessions,
        registration_map=args.registration_map,
        linked_out=args.linked_out,
        key_file=args.key,
        shifted_out=args.shifted_out,
        margin_hours=args.margin_hours,
        refresh_map=args.refresh_map,
    )


if __name__ == "__main__":
    main()
