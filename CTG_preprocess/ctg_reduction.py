from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from config import (
    DEFAULT_PARQUET_PATHS,
    DEFAULT_PARTITION_BUCKETS,
    DEFAULT_PARTITION_COLUMNS,
    DEFAULT_PARTITION_OUTPUT_DIR,
    DEFAULT_PARTITION_REPORT_EVERY,
    DEFAULT_REDUCTION_ROOT,
    DEFAULT_STAGE1_CUTOFF_DATE,
    DEFAULT_STAGE1_DIR,
    DEFAULT_STAGE2_DIR,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE3_OUTPUT_FILE,
    DEFAULT_STAGE3_GAP_MINUTES,
    DEFAULT_STAGE3_PREG_GAP_DAYS,
    DEFAULT_STAGE3_LAST_HOUR_MINUTES,
    DEFAULT_BABYID_SALT,
    DEFAULT_STAGE3_BUCKETS,
)
from partition_ctg import partition_ctg


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def stage1_timefilter(
    input_paths: list[str | Path],
    output_dir: str | Path,
    cutoff_date: str,
    batch_size: int = 65536,
    report_every_batches: int = DEFAULT_PARTITION_REPORT_EVERY,
) -> None:
    dataset = ds.dataset(input_paths, format="parquet")
    cutoff_dt = _parse_date(cutoff_date)
    filter_expr = ds.field("Timestamp") >= cutoff_dt
    scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)

    def batch_iter():
        # simple progress counter
        batches = 0
        rows = 0
        for batch in scanner.to_batches():
            batches += 1
            rows += batch.num_rows
            if report_every_batches and batches % report_every_batches == 0:
                print(f"Stage1: {batches} batches, {rows} rows")
            yield batch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds.write_dataset(
        batch_iter(),
        output_dir,
        format="parquet",
        existing_data_behavior="overwrite_or_ignore",
        schema=dataset.schema,
    )


def _compute_fhr(batch: pa.RecordBatch) -> pa.Array:
    h_cols = ["Hr1_0", "Hr1_1", "Hr1_2", "Hr1_3"]
    sums = None
    counts = None
    for name in h_cols:
        col = pc.fill_null(batch.column(batch.schema.get_field_index(name)), 0)
        mask = pc.greater(col, 0)
        val = pc.if_else(mask, col, 0)
        count = pc.cast(mask, pa.int32())
        sums = val if sums is None else pc.add(sums, val)
        counts = count if counts is None else pc.add(counts, count)
    has_vals = pc.greater(counts, 0)
    safe_counts = pc.if_else(has_vals, counts, 1)
    fhr = pc.if_else(has_vals, pc.divide(sums, safe_counts), 0)
    return pc.cast(fhr, pa.float32())


def _compute_toco(batch: pa.RecordBatch) -> pa.Array:
    idx = batch.schema.get_field_index("Toco_Values")
    if idx == -1:
        return pa.array([0.0] * batch.num_rows, type=pa.float32())
    toco_vals = batch.column(idx)

    decoded = None
    if hasattr(pc, "binary_base64_decode"):
        decoded = pc.binary_base64_decode(toco_vals)
    elif hasattr(pc, "binary_from_base64"):
        decoded = pc.binary_from_base64(toco_vals)

    if decoded is not None:
        data = decoded.to_pylist()
    else:
        data = []
        for v in toco_vals.to_pylist():
            if v is None:
                data.append(None)
                continue
            try:
                data.append(base64.b64decode(v))
            except Exception:
                data.append(None)

    out = []
    for v in data:
        if v is None:
            out.append(0.0)
            continue
        vals = [b for b in v]
        valid = [b for b in vals if 1 <= b <= 99]
        if valid:
            out.append(sum(valid) / len(valid))
        else:
            out.append(sum(vals) / len(vals) if vals else 0.0)
    return pa.array(out, type=pa.float32())


def stage2_columnfilter(
    input_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 65536,
    report_every_batches: int = DEFAULT_PARTITION_REPORT_EVERY,
) -> None:
    dataset = ds.dataset(str(input_dir), format="parquet")
    columns = [
        "Timestamp",
        "PatientID",
        "RegistrationID",
        "Hr1_0",
        "Hr1_1",
        "Hr1_2",
        "Hr1_3",
        "Toco_Values",
    ]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)

    reg_field = dataset.schema.field("RegistrationID") if "RegistrationID" in dataset.schema.names else None
    reg_type = reg_field.type if reg_field is not None else pa.string()

    schema = pa.schema(
        [
            ("Timestamp", dataset.schema.field("Timestamp").type),
            ("PatientID", dataset.schema.field("PatientID").type),
            ("RegistrationID", reg_type),
            ("FHR", pa.float32()),
            ("toco", pa.float32()),
        ]
    )

    def batch_iter():
        batches = 0
        rows = 0
        for batch in scanner.to_batches():
            batches += 1
            rows += batch.num_rows
            if report_every_batches and batches % report_every_batches == 0:
                print(f"Stage2: {batches} batches, {rows} rows")

            timestamp = batch.column(batch.schema.get_field_index("Timestamp"))
            patient_id = batch.column(batch.schema.get_field_index("PatientID"))
            reg_idx = batch.schema.get_field_index("RegistrationID")
            registration_id = (
                batch.column(reg_idx) if reg_idx != -1 else pa.nulls(batch.num_rows)
            )

            fhr = _compute_fhr(batch)
            toco = _compute_toco(batch)

            yield pa.RecordBatch.from_arrays(
                [timestamp, patient_id, registration_id, fhr, toco],
                schema=schema,
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds.write_dataset(
        batch_iter(),
        output_dir,
        format="parquet",
        existing_data_behavior="overwrite_or_ignore",
        schema=schema,
    )


def stage3_sessionfilter(
    input_dir: str | Path,
    output_file: str | Path,
    gap_minutes: int = DEFAULT_STAGE3_GAP_MINUTES,
    preg_gap_days: int = DEFAULT_STAGE3_PREG_GAP_DAYS,
    last_hour_minutes: int = DEFAULT_STAGE3_LAST_HOUR_MINUTES,
    babyid_salt: str = DEFAULT_BABYID_SALT,
    show_progress: bool = True,
    bucket_count: int = DEFAULT_STAGE3_BUCKETS,
    bucket_index: int | None = None,
) -> None:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for stage3. Install it with pip/uv.") from exc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

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
    safe_path = str(input_dir).replace("'", "''")
    con.execute(f"CREATE VIEW ctg AS SELECT * FROM read_parquet('{safe_path}')")

    def _pick_hash_func() -> str:
        for func in ("sha256", "md5"):
            try:
                con.execute(f"SELECT {func}('test')").fetchone()
                return func
            except Exception:
                continue
        return "md5"

    hash_func = _pick_hash_func()
    salt = babyid_salt.replace("'", "''")
    babyid_expr = (
        f"{hash_func}(concat('{salt}', '|', CAST(PatientID AS VARCHAR),"
        f" '|', CAST(session_end AS VARCHAR)))"
    )

    def _build_query(extra_where: str) -> str:
        where_clause = ("\n    WHERE " + extra_where) if extra_where else ""
        return f"""
WITH ordered AS (
    SELECT
        PatientID,
        RegistrationID,
        Timestamp,
        FHR,
        toco,
        Timestamp - LAG(Timestamp) OVER (PARTITION BY PatientID ORDER BY Timestamp) AS gap
    FROM ctg{where_clause}
),
sessioned AS (
    SELECT
        *,
        SUM(CASE WHEN gap IS NULL OR gap > INTERVAL '{gap_minutes} minutes'
            THEN 1 ELSE 0 END
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
        session_id,
        session_end,
        SUM(CASE WHEN prev_end IS NULL OR session_end - prev_end > INTERVAL '{preg_gap_days} days'
            THEN 1 ELSE 0 END
        ) OVER (PARTITION BY PatientID ORDER BY session_end) AS pregnancy_id
    FROM (
        SELECT
            *,
            LAG(session_end) OVER (PARTITION BY PatientID ORDER BY session_end) AS prev_end
        FROM session_end
    )
),
final_sessions AS (
    SELECT PatientID, pregnancy_id, MAX(session_end) AS session_end
    FROM preg_sessions
    GROUP BY PatientID, pregnancy_id
),
anchors AS (
    SELECT
        s.PatientID,
        p.pregnancy_id,
        s.session_id,
        p.session_end,
        MAX(s.Timestamp) FILTER (WHERE s.FHR > 0) AS last_nz_ts
    FROM sessioned s
    JOIN preg_sessions p
      ON s.PatientID = p.PatientID AND s.session_id = p.session_id
    JOIN final_sessions f
      ON p.PatientID = f.PatientID
     AND p.pregnancy_id = f.pregnancy_id
     AND p.session_end = f.session_end
    GROUP BY s.PatientID, p.pregnancy_id, s.session_id, p.session_end
),
final_rows AS (
    SELECT
        s.PatientID,
        s.RegistrationID,
        s.Timestamp,
        s.FHR,
        s.toco,
        a.session_end,
        COALESCE(a.last_nz_ts, a.session_end) AS anchor_ts
    FROM sessioned s
    JOIN anchors a
      ON s.PatientID = a.PatientID AND s.session_id = a.session_id
    WHERE s.Timestamp BETWEEN COALESCE(a.last_nz_ts, a.session_end)
        - INTERVAL '{last_hour_minutes} minutes'
        AND COALESCE(a.last_nz_ts, a.session_end)
)
SELECT
    {babyid_expr} AS BabyID,
    PatientID,
    RegistrationID,
    Timestamp,
    FHR,
    toco
FROM final_rows
"""

    def _bucket_expr() -> str:
        return f"(CAST(right(PatientID, 4) AS INTEGER) % {bucket_count})"

    output_path = Path(output_file)
    if bucket_count and bucket_count > 1:
        base_dir = output_path if output_path.suffix == "" else output_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_path.stem if output_path.suffix else "stage3_sessions"
        indices = [bucket_index] if bucket_index is not None else range(bucket_count)
        for idx in indices:
            out_path = base_dir / f"{prefix}_bucket_{idx:04d}.parquet"
            print(f"Stage3 bucket {idx+1}/{bucket_count}: {out_path.name}")
            if out_path.exists():
                out_path.unlink()
            query = _build_query(f"{_bucket_expr()} = {idx}")
            con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(out_path)])
    else:
        query = _build_query("")
        if output_path.exists():
            output_path.unlink()
        con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(output_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description="CTG reduction stages.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "stage2", "stage3", "partition"],
        required=True,
        help="Which stage to run.",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Input path(s) override for the stage.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory override for the stage.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=DEFAULT_STAGE1_CUTOFF_DATE,
        help="Stage1 cutoff date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Scanner batch size.",
    )
    parser.add_argument(
        "--report-every-batches",
        type=int,
        default=DEFAULT_PARTITION_REPORT_EVERY,
        help="Progress report frequency in batches (0 to disable).",
    )

    parser.add_argument(
        "--gap-minutes",
        type=int,
        default=DEFAULT_STAGE3_GAP_MINUTES,
        help="Stage3 session gap threshold (minutes).",
    )
    parser.add_argument(
        "--preg-gap-days",
        type=int,
        default=DEFAULT_STAGE3_PREG_GAP_DAYS,
        help="Stage3 pregnancy gap threshold (days).",
    )
    parser.add_argument(
        "--last-hour-minutes",
        type=int,
        default=DEFAULT_STAGE3_LAST_HOUR_MINUTES,
        help="Stage3 window length in minutes.",
    )
    parser.add_argument(
        "--babyid-salt",
        type=str,
        default=DEFAULT_BABYID_SALT,
        help="Salt used for BabyID hashing.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable DuckDB progress bar for stage3.",
    )

    parser.add_argument(
        "--bucket-count",
        type=int,
        default=DEFAULT_STAGE3_BUCKETS,
        help="Stage3: process in buckets (set to 1 to disable).",
    )
    parser.add_argument(
        "--bucket-index",
        type=int,
        default=None,
        help="Stage3: process a single bucket index (0..bucket-count-1).",
    )
    args = parser.parse_args()

    if args.stage == "stage1":
        stage1_timefilter(
            input_paths=args.input or DEFAULT_PARQUET_PATHS,
            output_dir=args.output or DEFAULT_STAGE1_DIR,
            cutoff_date=args.cutoff_date,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
        )
        return

    if args.stage == "stage2":
        stage2_columnfilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE1_DIR),
            output_dir=args.output or DEFAULT_STAGE2_DIR,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
        )
        return

    if args.stage == "stage3":
        stage3_sessionfilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE2_DIR),
            output_file=args.output or DEFAULT_STAGE3_OUTPUT_FILE,
            gap_minutes=args.gap_minutes,
            preg_gap_days=args.preg_gap_days,
            last_hour_minutes=args.last_hour_minutes,
            babyid_salt=args.babyid_salt,
            show_progress=not args.no_progress,
            bucket_count=args.bucket_count,
            bucket_index=args.bucket_index,
        )
        return

    if args.stage == "partition":
        partition_ctg(
            parquet_paths=[args.input[0]] if args.input else [DEFAULT_STAGE3_DIR],
            output_dir=args.output or DEFAULT_PARTITION_OUTPUT_DIR,
            cutoff_date=DEFAULT_STAGE1_CUTOFF_DATE,
            columns=DEFAULT_PARTITION_COLUMNS,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
            bucket_count=DEFAULT_PARTITION_BUCKETS,
        )
        return


if __name__ == "__main__":
    main()
