"""Attribute every registry (gravniva) birth row to exactly one Stage 7 match outcome.

The pipeline loses most registry-linked births before Stage 7 matching. This report replays
the Stage 7 registry cleaning (shared code with ``registry_matching.py``) and then walks each
registry birth row through the CTG stages to say *why* it did not end up matched:

``registry_row_excluded``
    Dropped by the Stage 7 registry cleaning (short personnummer, missing 5-min Apgar,
    missing birth day).
``no_ctg_for_patient``
    The mother's PatientID never appears in the Stage 3 output.
``ctg_only_outside_window``
    The mother has Stage 3 CTG pregnancies, but none anchored on the birth day or the day
    before. The signed day offset of the nearest pregnancy is recorded so the width of the
    +/-1 day window can be judged.
``dropped_stage4_duplicates``
    A Stage 3 BabyID sat in the window but was dropped by the Stage 4 duplicate filter.
``dropped_stage5_short_signal``
    Survived Stage 4 but was dropped by the Stage 5 minimum non-zero FHR filter.
``multiple_ctg_matches``
    More than one Stage 5.5 BabyID matches the row (Stage 7 drops such rows).
``ctg_shared_by_multiple_registry_rows``
    Exactly one BabyID matches, but that BabyID also matches another registry row
    (twins/multiples or duplicate registry rows); Stage 7 drops these too.
``matched``
    Ends up in the Stage 7 output.

Only counts are printed, never PatientIDs or BabyIDs.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
from cohort_report import _source_sql
from registry_matching import (
    UNIQUE_MATCHES_SQL,
    _count,
    _create_reg_clean_table,
    _create_reg_raw_view,
    _create_reg_table,
    _ctg_day_match_predicate,
)

from config import (
    DEFAULT_PATIENT_CSV,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE4_DIR,
    DEFAULT_STAGE5_5_OUTPUT_FILE,
)

CATEGORIES: tuple[str, ...] = (
    "registry_row_excluded",
    "no_ctg_for_patient",
    "ctg_only_outside_window",
    "dropped_stage4_duplicates",
    "dropped_stage5_short_signal",
    "multiple_ctg_matches",
    "ctg_shared_by_multiple_registry_rows",
    "matched",
)

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "registry_row_excluded": "dropped by Stage 7 registry cleaning (see sub-reasons)",
    "no_ctg_for_patient": "PatientID never appears in Stage 3 output",
    "ctg_only_outside_window": "Stage 3 CTG exists for the mother, none dated birth day or day before",
    "dropped_stage4_duplicates": "in-window Stage 3 BabyID removed by Stage 4 duplicate filter",
    "dropped_stage5_short_signal": "survived Stage 4, removed by Stage 5 minimum FHR filter",
    "multiple_ctg_matches": "more than one Stage 5.5 BabyID in the window (dropped by Stage 7)",
    "ctg_shared_by_multiple_registry_rows": (
        "its BabyID also matches another registry row, e.g. twins (dropped by Stage 7)"
    ),
    "matched": "present in Stage 7 output",
}

SUB_REASONS: tuple[str, ...] = ("short_personnummer", "missing_apgar5", "missing_birth_day")

# (label, inclusive low, inclusive high); ``None`` is open-ended. Offsets 0 and -1 are inside
# the Stage 7 window and therefore never occur in the outside-window group.
OFFSET_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("< -365", None, -366),
    ("-365..-31", -365, -31),
    ("-30..-8", -30, -8),
    ("-7..-2", -7, -2),
    ("+1", 1, 1),
    ("+2..+7", 2, 7),
    ("+8..+30", 8, 30),
    ("> +30", 31, None),
)

UNKNOWN_YEAR = "unknown"


def offset_bucket(offset_days: int) -> str:
    """Bucket label for a signed nearest-pregnancy offset (CTG anchor date minus birth day)."""
    for label, low, high in OFFSET_BUCKETS:
        if (low is None or offset_days >= low) and (high is None or offset_days <= high):
            return label
    return "in_window (unexpected)"


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _pct(part: int, whole: int) -> str:
    return f"{(part / whole * 100.0 if whole else 0.0):.2f}"


@dataclass
class MatchLossReport:
    registry_rows: int
    category_counts: dict[str, int]
    sub_reason_counts: dict[str, int]
    # Nearest Stage 3 pregnancy offset (days) -> rows, for the ``ctg_only_outside_window`` group.
    offset_counts: dict[int, int]
    # Birth year label -> category -> rows.
    year_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Stage 7 result recomputed from Stage 5.5 with the shared uniqueness SQL, for reconciliation.
    stage7_unique_matches: int = 0

    def offset_bucket_counts(self) -> dict[str, int]:
        counts = {label: 0 for label, _, _ in OFFSET_BUCKETS}
        for offset_days, rows in self.offset_counts.items():
            label = offset_bucket(offset_days)
            counts[label] = counts.get(label, 0) + rows
        return counts

    def to_markdown(self) -> str:
        total = self.registry_rows
        lines = ["# Registry match loss report", ""]
        lines.append(f"- Registry birth rows: {total}")
        lines.append(
            "- Stage 7 unique matches recomputed from Stage 5.5: "
            f"{self.stage7_unique_matches} ({_pct(self.stage7_unique_matches, total)} %)"
        )
        lines.append("")

        lines.append("## Loss attribution per registry birth row")
        lines.append("")
        lines.append("Each row is assigned the first category that applies, in this order.")
        lines.append("")
        lines.append("| Category | Sub-reason | Rows | % of registry | Meaning |")
        lines.append("|---|---|---:|---:|---|")
        for category in CATEGORIES:
            rows = self.category_counts.get(category, 0)
            lines.append(
                f"| {category} | | {rows} | {_pct(rows, total)} | "
                f"{CATEGORY_DESCRIPTIONS.get(category, '')} |"
            )
            if category == "registry_row_excluded":
                sub_labels = list(SUB_REASONS) + [
                    label for label in self.sub_reason_counts if label not in SUB_REASONS
                ]
                for sub in sub_labels:
                    sub_rows = self.sub_reason_counts.get(sub, 0)
                    lines.append(f"| {category} | {sub} | {sub_rows} | {_pct(sub_rows, total)} | |")
        lines.append(f"| Total | | {total} | {_pct(total, total)} | |")
        lines.append("")

        outside = self.category_counts.get("ctg_only_outside_window", 0)
        lines.append("## Nearest Stage 3 CTG pregnancy for `ctg_only_outside_window` rows")
        lines.append("")
        lines.append(
            "Offset = CTG anchor date (last timestamp of the Stage 3 window) minus registry "
            "birth day; negative means the CTG ended before the birth day. Stage 7 accepts "
            "offsets -1 and 0 only."
        )
        lines.append("")
        lines.append("| Offset (days) | Rows | % of group |")
        lines.append("|---|---:|---:|")
        for label, rows in self.offset_bucket_counts().items():
            lines.append(f"| {label} | {rows} | {_pct(rows, outside)} |")
        lines.append(f"| Total | {outside} | {_pct(outside, outside)} |")
        lines.append("")

        lines.append("## Registry birth rows by birth year")
        lines.append("")
        header = ["Year", "Rows", *CATEGORIES, "Matched %"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|---:|" + "---:|" * len(CATEGORIES) + "---:|")
        for year in sorted(self.year_counts, key=lambda y: (y == UNKNOWN_YEAR, y)):
            per_cat = self.year_counts[year]
            year_total = sum(per_cat.values())
            cells = [year, str(year_total)]
            cells.extend(str(per_cat.get(category, 0)) for category in CATEGORIES)
            cells.append(_pct(per_cat.get("matched", 0), year_total))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return "\n".join(lines)


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _stage_source(path: str | Path, label: str) -> str:
    source_sql, _, _ = _source_sql(path)
    if source_sql is None:
        raise FileNotFoundError(f"No parquet input found for {label}: {path}")
    return source_sql


def build_match_loss_table(
    con: duckdb.DuckDBPyConnection,
    registry_csv: str | Path,
    stage3_path: str | Path,
    stage4_path: str | Path,
    stage5_5_path: str | Path,
    verbose: bool = False,
) -> None:
    """Create the ``reg_loss`` temp table: one row per registry birth row with its category."""
    registry_csv = Path(registry_csv)
    if not registry_csv.exists():
        raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

    _log("Building registry tables (shared Stage 7 cleaning)...", verbose)
    _create_reg_raw_view(con, registry_csv)
    # Keep every registry row (Stage 7 itself never sees rows without a personnummer; here
    # they are counted under registry_row_excluded / short_personnummer).
    _create_reg_table(con, row_filter_sql="TRUE")
    _create_reg_clean_table(con)

    _log("Deriving per-BabyID CTG dates from Stage 3 and Stage 4...", verbose)
    for table, path, label in (
        ("s3_babies", stage3_path, "stage3"),
        ("s4_babies", stage4_path, "stage4"),
    ):
        source = _stage_source(path, label)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {table} AS
            SELECT
                BabyID,
                MIN(PatientID) AS PatientID,
                CAST(MAX(Timestamp) AS DATE) AS ctg_date
            FROM {source}
            GROUP BY BabyID
            """
        )

    _log("Loading Stage 5.5 BabyID map...", verbose)
    s55_source = _stage_source(stage5_5_path, "stage5_5")
    s55_cols = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {s55_source}").fetchall()}
    if "ctg_date" in s55_cols:
        # Same map Stage 7 builds from the sorted Stage 5.5 output.
        s55_map_sql = f"SELECT DISTINCT BabyID, PatientID, ctg_date FROM {s55_source}"
    else:
        s55_map_sql = f"""
            SELECT BabyID, MIN(PatientID) AS PatientID, CAST(MAX(Timestamp) AS DATE) AS ctg_date
            FROM {s55_source}
            GROUP BY BabyID
        """
    con.execute(f"CREATE OR REPLACE TEMP TABLE s55_map AS {s55_map_sql}")

    _log("Replaying Stage 7 matching on Stage 5.5...", verbose)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE matches AS
        SELECT r.reg_row, r.PatientID, r.birth_day, m.BabyID, m.ctg_date
        FROM reg_clean r
        JOIN s55_map m
          ON r.PatientID = m.PatientID
         AND {_ctg_day_match_predicate("m.ctg_date", "r.birth_day")}
        """
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE unique_matches AS {UNIQUE_MATCHES_SQL}")

    _log("Classifying registry rows...", verbose)
    s3_offset = "date_diff('day', r.birth_day, s3.ctg_date)"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE reg_loss AS
        WITH s3_patients AS (
            SELECT DISTINCT PatientID FROM s3_babies
        ),
        in_window AS (
            SELECT
                r.reg_row,
                COUNT(*) AS n_stage3,
                COUNT(s4.BabyID) AS n_stage4
            FROM reg_clean r
            JOIN s3_babies s3
              ON s3.PatientID = r.PatientID
             AND {_ctg_day_match_predicate("s3.ctg_date", "r.birth_day")}
            LEFT JOIN s4_babies s4 ON s4.BabyID = s3.BabyID
            GROUP BY r.reg_row
        ),
        stage55 AS (
            SELECT reg_row, COUNT(*) AS n_stage55
            FROM matches
            GROUP BY reg_row
        ),
        unique_rows AS (
            SELECT DISTINCT reg_row FROM unique_matches
        ),
        nearest AS (
            SELECT reg_row, offset_days
            FROM (
                SELECT
                    r.reg_row,
                    {s3_offset} AS offset_days,
                    row_number() OVER (
                        PARTITION BY r.reg_row ORDER BY abs({s3_offset}), {s3_offset}
                    ) AS rn
                FROM reg_clean r
                JOIN s3_babies s3 ON s3.PatientID = r.PatientID
            )
            WHERE rn = 1
        ),
        classified AS (
            SELECT
                g.reg_row,
                g.birth_day,
                CASE
                    WHEN c.reg_row IS NULL THEN 'registry_row_excluded'
                    WHEN p.PatientID IS NULL THEN 'no_ctg_for_patient'
                    WHEN COALESCE(w.n_stage3, 0) = 0 THEN 'ctg_only_outside_window'
                    WHEN COALESCE(w.n_stage4, 0) = 0 THEN 'dropped_stage4_duplicates'
                    WHEN COALESCE(m.n_stage55, 0) = 0 THEN 'dropped_stage5_short_signal'
                    WHEN m.n_stage55 > 1 THEN 'multiple_ctg_matches'
                    WHEN u.reg_row IS NULL THEN 'ctg_shared_by_multiple_registry_rows'
                    ELSE 'matched'
                END AS category,
                CASE
                    WHEN c.reg_row IS NOT NULL THEN NULL
                    WHEN g.reg_digits IS NULL OR length(g.reg_digits) < 12 THEN 'short_personnummer'
                    WHEN g.apgar5 IS NULL THEN 'missing_apgar5'
                    WHEN g.birth_day IS NULL THEN 'missing_birth_day'
                    ELSE 'other'
                END AS sub_reason,
                n.offset_days
            FROM reg g
            LEFT JOIN reg_clean c ON c.reg_row = g.reg_row
            LEFT JOIN s3_patients p ON p.PatientID = c.PatientID
            LEFT JOIN in_window w ON w.reg_row = g.reg_row
            LEFT JOIN stage55 m ON m.reg_row = g.reg_row
            LEFT JOIN unique_rows u ON u.reg_row = g.reg_row
            LEFT JOIN nearest n ON n.reg_row = g.reg_row
        )
        SELECT
            reg_row,
            birth_day,
            category,
            sub_reason,
            CASE WHEN category = 'ctg_only_outside_window' THEN offset_days END
                AS nearest_offset_days
        FROM classified
        """
    )


def summarize_match_loss(con: duckdb.DuckDBPyConnection) -> MatchLossReport:
    """Aggregate the ``reg_loss`` table into counts only (no identifiers)."""
    category_counts = {
        str(category): int(rows)
        for category, rows in con.execute(
            "SELECT category, COUNT(*) FROM reg_loss GROUP BY category"
        ).fetchall()
    }
    sub_reason_counts = {
        str(sub): int(rows)
        for sub, rows in con.execute(
            """
            SELECT sub_reason, COUNT(*)
            FROM reg_loss
            WHERE category = 'registry_row_excluded'
            GROUP BY sub_reason
            """
        ).fetchall()
    }
    offset_counts = {
        int(offset_days): int(rows)
        for offset_days, rows in con.execute(
            """
            SELECT nearest_offset_days, COUNT(*)
            FROM reg_loss
            WHERE category = 'ctg_only_outside_window' AND nearest_offset_days IS NOT NULL
            GROUP BY nearest_offset_days
            """
        ).fetchall()
    }
    year_counts: dict[str, dict[str, int]] = {}
    for year, category, rows in con.execute(
        """
        SELECT CAST(EXTRACT(YEAR FROM birth_day) AS INTEGER) AS year, category, COUNT(*)
        FROM reg_loss
        GROUP BY year, category
        """
    ).fetchall():
        label = UNKNOWN_YEAR if year is None else str(int(year))
        year_counts.setdefault(label, {})[str(category)] = int(rows)

    return MatchLossReport(
        registry_rows=_count(con, "SELECT COUNT(*) FROM reg_loss"),
        category_counts=category_counts,
        sub_reason_counts=sub_reason_counts,
        offset_counts=offset_counts,
        year_counts=year_counts,
        stage7_unique_matches=_count(con, "SELECT COUNT(*) FROM unique_matches"),
    )


def compute_match_loss_report(
    registry_csv: str | Path,
    stage3_path: str | Path,
    stage4_path: str | Path,
    stage5_5_path: str | Path,
    show_progress: bool = True,
    verbose: bool = False,
) -> MatchLossReport:
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
    try:
        build_match_loss_table(
            con, registry_csv, stage3_path, stage4_path, stage5_5_path, verbose=verbose
        )
        return summarize_match_loss(con)
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute every registry birth row to one Stage 7 match-loss reason."
    )
    parser.add_argument("--registry-csv", type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--stage3", type=str, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--stage4", type=str, default=DEFAULT_STAGE4_DIR)
    parser.add_argument("--stage5-5", type=str, default=DEFAULT_STAGE5_5_OUTPUT_FILE)
    parser.add_argument("--out", type=str, default=None, help="Also write the markdown here.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    report = compute_match_loss_report(
        registry_csv=args.registry_csv,
        stage3_path=args.stage3,
        stage4_path=args.stage4,
        stage5_5_path=args.stage5_5,
        show_progress=not args.no_progress,
        verbose=True,
    )
    markdown = report.to_markdown()
    print(markdown)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown + "\n", encoding="utf-8")
        _log(f"Wrote {out_path}", True)
    _log(f"Runtime: {_fmt_seconds(time.perf_counter() - start)}", True)


if __name__ == "__main__":
    main()
