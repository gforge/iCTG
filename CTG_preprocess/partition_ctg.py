from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from config import (
    DEFAULT_PARQUET_PATHS,
    DEFAULT_PARTITION_COLUMNS,
    DEFAULT_PARTITION_CUTOFF_DATE,
    DEFAULT_PARTITION_OUTPUT_DIR,
    DEFAULT_PARTITION_REPORT_EVERY,
    DEFAULT_PARTITION_BUCKETS,
)


def _parse_cutoff_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def partition_ctg(
    parquet_paths: list[str | Path],
    output_dir: str | Path,
    cutoff_date: str,
    columns: list[str],
    batch_size: int = 65536,
    report_every_batches: int = DEFAULT_PARTITION_REPORT_EVERY,
    max_batches: int = 0,
    bucket_count: int = DEFAULT_PARTITION_BUCKETS,
) -> None:
    dataset = ds.dataset(parquet_paths, format="parquet")

    cutoff_dt = _parse_cutoff_date(cutoff_date)
    filter_expr = ds.field("Timestamp") >= cutoff_dt

    scanner = dataset.scanner(columns=columns, filter=filter_expr, batch_size=batch_size)
    base_fields = [dataset.schema.field(name) for name in columns]
    schema = pa.schema(base_fields + [
        pa.field("ctg_date", pa.date32()),
        pa.field("patient_bucket", pa.int16()),
    ])

    def batch_iter():
        start_time = time.perf_counter()
        batches = 0
        rows = 0
        for batch in scanner.to_batches():
            batches += 1
            rows += batch.num_rows
            if report_every_batches and batches % report_every_batches == 0:
                elapsed = time.perf_counter() - start_time
                rate = rows / elapsed if elapsed else 0.0
                print(f"Processed {batches} batches, {rows} rows ({rate:,.0f} rows/s)")
            ts = batch.column(batch.schema.get_field_index("Timestamp"))
            ctg_date = pc.cast(ts, pa.date32())
            pid = batch.column(batch.schema.get_field_index("PatientID"))
            last4 = pc.utf8_slice_codeunits(pid, 9, 13)
            pid_int = pc.cast(last4, pa.int32())
            if hasattr(pc, "mod"):
                bucket = pc.mod(pid_int, bucket_count)
            elif hasattr(pc, "modulus"):
                bucket = pc.modulus(pid_int, bucket_count)
            else:
                raise RuntimeError("pyarrow compute missing mod/modulus; upgrade pyarrow")
            bucket = pc.fill_null(bucket, -1)
            bucket = pc.cast(bucket, pa.int16())
            batch = batch.append_column("ctg_date", ctg_date)
            batch = batch.append_column("patient_bucket", bucket)
            yield batch
            if max_batches and batches >= max_batches:
                break
        if report_every_batches:
            elapsed = time.perf_counter() - start_time
            rate = rows / elapsed if elapsed else 0.0
            print(f"Done: {batches} batches, {rows} rows ({rate:,.0f} rows/s)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds.write_dataset(
        batch_iter(),
        output_dir,
        format="parquet",
        partitioning=["ctg_date", "patient_bucket"],
        existing_data_behavior="overwrite_or_ignore",
        schema=schema,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition CTG parquet dataset by date.")
    parser.add_argument(
        "--parquet",
        type=str,
        nargs="+",
        default=DEFAULT_PARQUET_PATHS,
        help="Input parquet files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_PARTITION_OUTPUT_DIR,
        help="Output directory for the partitioned dataset.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=DEFAULT_PARTITION_CUTOFF_DATE,
        help="Drop rows before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=DEFAULT_PARTITION_COLUMNS,
        help="Columns to keep in the partitioned dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Scanner batch size.",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=DEFAULT_PARTITION_BUCKETS,
        help="Number of patient buckets for partitioning.",
    )
    parser.add_argument(
        "--report-every-batches",
        type=int,
        default=DEFAULT_PARTITION_REPORT_EVERY,
        help="Print progress every N batches (0 to disable).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after N batches (0 for no limit).",
    )
    args = parser.parse_args()

    partition_ctg(
        parquet_paths=args.parquet,
        output_dir=args.output_dir,
        cutoff_date=args.cutoff_date,
        columns=args.columns,
        batch_size=args.batch_size,
        report_every_batches=args.report_every_batches,
        max_batches=args.max_batches,
        bucket_count=args.bucket_count,
    )


if __name__ == "__main__":
    main()
