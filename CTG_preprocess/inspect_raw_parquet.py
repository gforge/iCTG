from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from config import DEFAULT_PARQUET_PATHS


def _open_dataset(parquet_paths: Iterable[str | Path]) -> tuple[ds.Dataset, list[str]]:
    dataset = ds.dataset(parquet_paths, format="parquet")
    files = list(dataset.files)
    if not files:
        raise FileNotFoundError("No parquet files found for the provided paths.")
    return dataset, files


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


def _count_rows(files: Iterable[str]) -> tuple[int, list[tuple[str, int]]]:
    total_rows = 0
    per_file = []
    for path in files:
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows if pf.metadata else 0
        per_file.append((path, rows))
        total_rows += rows
    return total_rows, per_file


def _count_unique_values(
    dataset: ds.Dataset,
    column: str,
    batch_size: int,
    report_every_batches: int,
) -> int:
    scanner = dataset.scanner(columns=[column], batch_size=batch_size, use_threads=True)
    seen: set[object] = set()
    start_time = time.perf_counter()
    rows = 0
    batches = 0
    for batch in scanner.to_batches():
        batches += 1
        rows += batch.num_rows
        arr = batch.column(0)
        if arr.null_count:
            arr = arr.drop_null()
        values = arr.to_pylist()
        if values:
            seen.update(values)
        if report_every_batches and batches % report_every_batches == 0:
            elapsed = time.perf_counter() - start_time
            rate = rows / elapsed if elapsed else 0.0
            print(
                f"{column}: {batches} batches, {rows:,} rows "
                f"({rate:,.0f} rows/s), unique so far {len(seen):,}"
            )
    return len(seen)


def _check_registration_mapping(
    dataset: ds.Dataset,
    patient_col: str,
    registration_col: str,
    batch_size: int,
    report_every_batches: int,
    max_conflict_samples: int,
    stop_after_conflicts: int,
) -> tuple[bool, int, dict[object, set[object]]]:
    scanner = dataset.scanner(
        columns=[patient_col, registration_col],
        batch_size=batch_size,
        use_threads=True,
    )
    mapping: dict[object, object] = {}
    conflicts: dict[object, set[object]] = {}
    conflict_count = 0
    start_time = time.perf_counter()
    rows = 0
    batches = 0
    for batch in scanner.to_batches():
        batches += 1
        rows += batch.num_rows
        patient_vals = batch.column(0).to_pylist()
        registration_vals = batch.column(1).to_pylist()
        for patient_id, registration_id in zip(patient_vals, registration_vals):
            if patient_id is None or registration_id is None:
                continue
            existing = mapping.get(registration_id)
            if existing is None:
                mapping[registration_id] = patient_id
                continue
            if existing != patient_id:
                if registration_id not in conflicts:
                    conflict_count += 1
                if len(conflicts) < max_conflict_samples:
                    conflicts.setdefault(registration_id, set()).update(
                        [existing, patient_id]
                    )
                if stop_after_conflicts and conflict_count >= stop_after_conflicts:
                    return False, conflict_count, conflicts
        if report_every_batches and batches % report_every_batches == 0:
            elapsed = time.perf_counter() - start_time
            rate = rows / elapsed if elapsed else 0.0
            print(
                f"Mapping check: {batches} batches, {rows:,} rows "
                f"({rate:,.0f} rows/s), conflicts {conflict_count:,}"
            )
    return conflict_count == 0, conflict_count, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw CTG parquet exports and compute row/unique counts."
        )
    )
    parser.add_argument(
        "--parquet",
        type=str,
        nargs="+",
        default=DEFAULT_PARQUET_PATHS,
        help="Input parquet files or directories.",
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
        default="RegistrationID",
        help="Registration ID column name.",
    )
    parser.add_argument(
        "--registration-col-fallback",
        type=str,
        default="RegistratonID",
        help="Fallback registration column name if the preferred name is missing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Scanner batch size for unique counts/mapping checks.",
    )
    parser.add_argument(
        "--report-every-batches",
        type=int,
        default=0,
        help="Print progress every N batches (0 to disable).",
    )
    parser.add_argument(
        "--check-registration-mapping",
        action="store_true",
        help="Validate that each registration ID maps to a single patient ID.",
    )
    parser.add_argument(
        "--max-conflict-samples",
        type=int,
        default=5,
        help="Max number of conflicting registration IDs to sample for output.",
    )
    parser.add_argument(
        "--stop-after-conflicts",
        type=int,
        default=0,
        help="Stop early after N conflicting registration IDs (0 for full scan).",
    )
    parser.add_argument(
        "--show-file-counts",
        action="store_true",
        help="Print per-file row counts.",
    )
    args = parser.parse_args()

    dataset, files = _open_dataset(args.parquet)
    patient_col = _resolve_column_name(dataset.schema.names, args.patient_col)
    registration_col = _resolve_column_name(
        dataset.schema.names,
        args.registration_col,
        args.registration_col_fallback,
    )

    print(f"Files: {len(files)}")
    total_rows, per_file = _count_rows(files)
    print(f"Total rows: {total_rows:,}")
    if args.show_file_counts:
        for path, rows in per_file:
            print(f"  {path}: {rows:,}")

    unique_patient_ids = _count_unique_values(
        dataset,
        patient_col,
        batch_size=args.batch_size,
        report_every_batches=args.report_every_batches,
    )
    print(f"Unique {patient_col}: {unique_patient_ids:,}")

    unique_registration_ids = _count_unique_values(
        dataset,
        registration_col,
        batch_size=args.batch_size,
        report_every_batches=args.report_every_batches,
    )
    print(f"Unique {registration_col}: {unique_registration_ids:,}")

    if args.check_registration_mapping:
        ok, conflict_count, conflicts = _check_registration_mapping(
            dataset,
            patient_col=patient_col,
            registration_col=registration_col,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
            max_conflict_samples=args.max_conflict_samples,
            stop_after_conflicts=args.stop_after_conflicts,
        )
        if ok:
            print(
                "RegistrationID mapping: OK (each registration maps to one patient)."
            )
        else:
            print(
                "RegistrationID mapping: FAIL "
                f"({conflict_count:,} conflicting registrations found)."
            )
            for reg_id, patient_ids in conflicts.items():
                sample = ", ".join(str(pid) for pid in sorted(patient_ids))
                print(f"  {reg_id}: {sample}")


if __name__ == "__main__":
    main()
