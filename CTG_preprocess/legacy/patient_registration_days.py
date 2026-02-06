from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from config import DEFAULT_PARQUET_PATHS
from ctg_processing import format_patient_id


def _open_dataset(
    parquet_paths: Iterable[str | Path],
    dataset_dir: str | Path | None,
) -> ds.Dataset:
    if dataset_dir:
        return ds.dataset(dataset_dir, format="parquet")
    return ds.dataset(parquet_paths, format="parquet")


def _resolve_column_name(
    schema_names: list[str],
    preferred: str,
    fallback: str | None = None,
) -> str:
    if preferred in schema_names:
        return preferred
    if fallback and fallback in schema_names:
        return fallback
    candidates = ", ".join(schema_names)
    if fallback:
        raise ValueError(
            f"Column not found: '{preferred}' or '{fallback}'. Available: {candidates}"
        )
    raise ValueError(f"Column not found: '{preferred}'. Available: {candidates}")


def _to_date_array(values: pa.Array) -> pa.Array:
    try:
        return pc.cast(values, pa.date32())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        # Fallback: convert to python objects and normalize to date.
        out: list[date | None] = []
        for value in values.to_pylist():
            if value is None:
                out.append(None)
            elif isinstance(value, date):
                out.append(value)
            else:
                try:
                    out.append(date.fromisoformat(str(value)[:10]))
                except ValueError:
                    out.append(None)
        return pa.array(out, type=pa.date32())


def _scan_patient_days(
    dataset: ds.Dataset,
    patient_id: str,
    patient_col: str,
    timestamp_col: str,
    registration_col: str,
    batch_size: int,
    report_every_batches: int,
) -> dict[date, set[object]]:
    filter_expr = ds.field(patient_col) == patient_id
    scanner = dataset.scanner(
        columns=[timestamp_col, registration_col],
        filter=filter_expr,
        batch_size=batch_size,
        use_threads=True,
    )

    day_map: dict[date, set[object]] = {}
    batches = 0
    rows = 0
    start_time = time.perf_counter()

    for batch in scanner.to_batches():
        batches += 1
        rows += batch.num_rows

        dates = _to_date_array(batch.column(0)).to_pylist()
        regs = batch.column(1).to_pylist()

        for day, reg in zip(dates, regs):
            if day is None or reg is None:
                continue
            day_map.setdefault(day, set()).add(reg)

        if report_every_batches and batches % report_every_batches == 0:
            elapsed = time.perf_counter() - start_time
            rate = rows / elapsed if elapsed else 0.0
            print(
                f"Scanned {batches} batches, {rows:,} rows "
                f"({rate:,.0f} rows/s)"
            )

    return day_map


def _print_summary(day_map: dict[date, set[object]]) -> None:
    if not day_map:
        print("No rows found for the patient.")
        return

    unique_regs = set()
    multi_reg_days = []
    for day, regs in day_map.items():
        unique_regs.update(regs)
        if len(regs) > 1:
            multi_reg_days.append(day)

    print(f"Days with CTG data: {len(day_map):,}")
    print(f"Unique RegistrationIDs: {len(unique_regs):,}")
    print(f"Days with multiple RegistrationIDs: {len(multi_reg_days):,}")

    for day in sorted(day_map):
        regs = ", ".join(str(r) for r in sorted(day_map[day]))
        print(f"{day.isoformat()}: {regs}")

    if multi_reg_days:
        print("Days with multiple RegistrationIDs (potential conflicts):")
        for day in sorted(multi_reg_days):
            regs = ", ".join(str(r) for r in sorted(day_map[day]))
            print(f"  {day.isoformat()}: {regs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show per-day RegistrationIDs for a single PatientID."
        )
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        required=True,
        help="PatientID to inspect.",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        nargs="+",
        default=DEFAULT_PARQUET_PATHS,
        help="Input parquet files or directories (ignored if --dataset-dir is set).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional dataset directory (e.g. partitioned output).",
    )
    parser.add_argument(
        "--patient-col",
        type=str,
        default="PatientID",
        help="Patient ID column name.",
    )
    parser.add_argument(
        "--registration-col",
        type=str,
        default="RegistriationID",
        help="Registration ID column name.",
    )
    parser.add_argument(
        "--registration-col-fallback",
        type=str,
        default="RegistrationID",
        help="Fallback registration column name if preferred is missing.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="Timestamp",
        help="Timestamp column name.",
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
        default=0,
        help="Print progress every N batches (0 to disable).",
    )
    args = parser.parse_args()

    patient_id = format_patient_id(args.patient_id)
    if not patient_id:
        raise ValueError("PatientID is empty after formatting.")

    dataset = _open_dataset(args.parquet, args.dataset_dir)
    schema_names = dataset.schema.names
    patient_col = _resolve_column_name(schema_names, args.patient_col)
    timestamp_col = _resolve_column_name(schema_names, args.timestamp_col)
    registration_col = _resolve_column_name(
        schema_names,
        args.registration_col,
        args.registration_col_fallback,
    )

    print(f"PatientID: {patient_id}")
    if args.dataset_dir:
        print(f"Dataset dir: {args.dataset_dir}")

    day_map = _scan_patient_days(
        dataset,
        patient_id=patient_id,
        patient_col=patient_col,
        timestamp_col=timestamp_col,
        registration_col=registration_col,
        batch_size=args.batch_size,
        report_every_batches=args.report_every_batches,
    )
    _print_summary(day_map)


if __name__ == "__main__":
    main()
