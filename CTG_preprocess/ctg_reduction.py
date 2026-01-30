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
    fhr = pc.if_else(has_vals, pc.divide(sums, counts), 0)
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


def stage3_sessionfilter(*_args, **_kwargs) -> None:
    raise NotImplementedError("Stage 3 (session filter) is not implemented yet.")


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
        stage3_sessionfilter()
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
