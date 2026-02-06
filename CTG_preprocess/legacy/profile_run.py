from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import (
    DEFAULT_DOWNSAMPLE_MODE,
    DEFAULT_PARQUET_PATHS,
    DEFAULT_PARTITION_BUCKETS,
    DEFAULT_PARTITION_OUTPUT_DIR,
    DEFAULT_PATIENT_CSV,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_USE_PARTITIONED_DATASET,
)
from ctg_processing import filter_ctg_data, load_ctg_data
from main import iter_patients, _build_dataset


def profile_run(
    patient_csv: str | Path,
    parquet_paths: list[str | Path],
    start: int,
    limit: int,
    sample_rate: int,
    downsample_mode: str,
) -> None:
    dataset = _build_dataset(parquet_paths)

    totals = {
        "patients": 0,
        "load_ctg_s": 0.0,
        "filter_s": 0.0,
        "skipped_no_apgar": 0,
        "skipped_no_ctg": 0,
        "skipped_filtered": 0,
        "valid": 0,
    }

    start_total = time.perf_counter()
    for patient_index, patient in iter_patients(patient_csv, start, limit):
        totals["patients"] += 1
        apgar5 = patient.get("apgar5")
        if apgar5 is None:
            totals["skipped_no_apgar"] += 1
            continue

        load_start = time.perf_counter()
        ctg_df = load_ctg_data(
            dataset,
            patient.get("pn", ""),
            patient.get("birth_day"),
            sample_rate_hz=sample_rate,
            downsample_mode=downsample_mode,
            bucket_count=DEFAULT_PARTITION_BUCKETS,
            partition_root=DEFAULT_PARTITION_OUTPUT_DIR if DEFAULT_USE_PARTITIONED_DATASET else None,
        )
        totals["load_ctg_s"] += time.perf_counter() - load_start
        if ctg_df is None:
            totals["skipped_no_ctg"] += 1
            continue

        filter_start = time.perf_counter()
        filtered_df = filter_ctg_data(ctg_df, sample_rate_hz=sample_rate)
        totals["filter_s"] += time.perf_counter() - filter_start
        if filtered_df is None:
            totals["skipped_filtered"] += 1
            continue

        totals["valid"] += 1

    elapsed = time.perf_counter() - start_total
    print(f"Profiled {totals['patients']} patients in {elapsed:.2f}s")
    if totals["patients"]:
        print(f"Average wall time per patient: {elapsed / totals['patients']:.2f}s")
    print(
        f"valid={totals['valid']}, skipped_no_apgar={totals['skipped_no_apgar']}, "
        f"skipped_no_ctg={totals['skipped_no_ctg']}, skipped_filtered={totals['skipped_filtered']}"
    )
    print(
        f"Load CTG time: {totals['load_ctg_s']:.2f}s, "
        f"Filter time: {totals['filter_s']:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CTG preprocessing steps.")
    parser.add_argument("--patient-csv", type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--parquet", type=str, nargs="+", default=DEFAULT_PARQUET_PATHS)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, choices=[1, 4], default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--downsample-mode", type=str, choices=["mean", "first"], default=DEFAULT_DOWNSAMPLE_MODE)
    args = parser.parse_args()

    profile_run(
        patient_csv=args.patient_csv,
        parquet_paths=args.parquet,
        start=args.start,
        limit=args.limit,
        sample_rate=args.sample_rate,
        downsample_mode=args.downsample_mode,
    )


if __name__ == "__main__":
    main()
