from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from config import (
    DEFAULT_DOWNSAMPLE_MODE,
    DEFAULT_MAX_BATCH_SIZE_GB,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARQUET_PATHS,
    DEFAULT_PARTITION_OUTPUT_DIR,
    DEFAULT_PARTITION_BUCKETS,
    DEFAULT_USE_PARTITIONED_DATASET,
    DEFAULT_PATIENT_CSV,
    DEFAULT_REPORT_EVERY,
    DEFAULT_SAMPLE_RATE_HZ,
)
from ctg_processing import filter_ctg_data, load_ctg_data

DEFAULT_CTGSOURCE_PATHS = ([DEFAULT_PARTITION_OUTPUT_DIR] if DEFAULT_USE_PARTITIONED_DATASET else DEFAULT_PARQUET_PATHS)

def _build_dataset(parquet_paths: str | Path | Iterable[str | Path]) -> ds.Dataset:
    if isinstance(parquet_paths, (str, Path)):
        return ds.dataset(str(parquet_paths), format="parquet")

    paths = [Path(p) for p in parquet_paths]
    if len(paths) == 1 and paths[0].is_dir():
        return ds.dataset(str(paths[0]), format="parquet")

    if any(p.is_dir() for p in paths):
        expanded: list[Path] = []
        for p in paths:
            if p.is_dir():
                expanded.extend(sorted(p.rglob("*.parquet")))
            else:
                expanded.append(p)
        return ds.dataset([str(p) for p in expanded], format="parquet")

    return ds.dataset([str(p) for p in paths], format="parquet")


def _detect_delimiter(sample: str) -> str:
    if sample.count(";") > sample.count(","):
        return ";"
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return ","


def _parse_optional_int(value: str) -> int | None:
    if not value:
        return None
    cleaned = value.strip().replace(",", ".")
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _parse_patient_row(row: dict[str, str]) -> dict[str, Any]:
    pn_raw = (row.get("personnummer_mor") or "").strip()
    birth_raw = (row.get("forlossningsdatum_fv1") or "").strip()
    apgar_raw = (row.get("apgar_5_min") or "").strip()

    birth_day = date.fromisoformat(birth_raw) if birth_raw else None
    apgar5 = _parse_optional_int(apgar_raw)

    return {
        "pn": pn_raw,
        "birth_day": birth_day,
        "apgar5": apgar5,
    }


def iter_patients(
    csv_path: str | Path, start: int, count: int
) -> Iterable[tuple[int, dict[str, Any]]]:
    csv_path = Path(csv_path)
    end = start + count - 1

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = _detect_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)

        for index, row in enumerate(reader, start=1):
            if index < start:
                continue
            if index > end:
                break
            yield index, _parse_patient_row(row)


def retrieve_patient(n: int, csv_path: str | Path) -> dict[str, Any]:
    """
    Return a patient's core fields from row n (1-based) in the CSV.

    Expected columns:
      - personnummer_mor (YYYYMMDDXXXX, kept as string)
      - forlossningsdatum_fv1 (YYYY-MM-DD, parsed to date)
      - apgar_5_min (int 1-10, or None if missing)
    """
    if n < 1:
        raise ValueError("n must be >= 1 (patient rows are 1-based)")

    csv_path = Path(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = _detect_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader, start=1):
            if index == n:
                return _parse_patient_row(row)

    raise IndexError(f"Patient row {n} not found in {csv_path}")


def _get_next_baby_id(metadata_path: Path) -> int:
    if not metadata_path.exists():
        return 1

    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        max_id = 0
        for row in reader:
            try:
                max_id = max(max_id, int(row.get("BabyID", 0)))
            except ValueError:
                continue

    return max_id + 1


def _append_metadata(metadata_path: Path, baby_id: int, birth_day: date | None, apgar5: int | None) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["BabyID", "birth_day", "apgar5"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "BabyID": baby_id,
                "birth_day": birth_day.isoformat() if birth_day else "",
                "apgar5": apgar5 if apgar5 is not None else "",
            }
        )


class _ParquetBatchWriter:
    def __init__(self, output_path: Path, append_existing: bool) -> None:
        self.output_path = output_path
        self.append_existing = append_existing
        self.temp_path = (
            output_path.with_suffix(".tmp.parquet") if append_existing else output_path
        )
        self.writer: pq.ParquetWriter | None = None

    def _ensure_writer(self, table: pa.Table) -> None:
        if self.writer is not None:
            return

        if self.append_existing and self.output_path.exists():
            parquet_file = pq.ParquetFile(self.output_path)
            self.writer = pq.ParquetWriter(self.temp_path, parquet_file.schema_arrow)
            for i in range(parquet_file.num_row_groups):
                self.writer.write_table(parquet_file.read_row_group(i))
            return

        self.writer = pq.ParquetWriter(self.temp_path, table.schema)

    def write(self, table: pa.Table) -> None:
        self._ensure_writer(table)
        if self.writer:
            self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        if self.append_existing:
            self.temp_path.replace(self.output_path)


def _dataset_output_path(output_dir: Path, batch_id: int, start_patient: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"ctg_batch_{batch_id:04d}_start_{start_patient}.parquet"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = output_dir / f"{base.stem}_part_{suffix}{base.suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def process_patients(
    patient_csv: str | Path,
    parquet_paths: list[str | Path],
    output_dir: str | Path,
    start_patient: int,
    patient_count: int,
    batch_id: int,
    min_data: float = 0.5,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE,
    report_every: int = DEFAULT_REPORT_EVERY,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    max_batch_size_gb: float = DEFAULT_MAX_BATCH_SIZE_GB,
) -> None:
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.csv"

    valid_patients = 0
    invalid_patients = 0

    next_baby_id = _get_next_baby_id(metadata_path)
    dataset = _build_dataset(parquet_paths)

    if output_mode == "append":
        output_path = output_dir / f"ctg_batch_{batch_id:04d}.parquet"
        if output_path.exists():
            max_bytes = max_batch_size_gb * 1024 * 1024 * 1024
            if output_path.stat().st_size >= max_bytes:
                output_mode = "dataset"

    if output_mode == "dataset":
        output_path = _dataset_output_path(output_dir, batch_id, start_patient)
        writer = _ParquetBatchWriter(output_path, append_existing=False)
    else:
        writer = _ParquetBatchWriter(output_path, append_existing=True)

    for patient_index, patient in iter_patients(patient_csv, start_patient, patient_count):

        apgar5 = patient.get("apgar5")
        if apgar5 is None:
            invalid_patients += 1
            continue

        pn = patient.get("pn", "")
        birth_day = patient.get("birth_day")

        ctg_df = load_ctg_data(
            dataset,
            pn,
            birth_day,
            sample_rate_hz=sample_rate_hz,
            downsample_mode=downsample_mode,
            bucket_count=DEFAULT_PARTITION_BUCKETS,
        )
        if ctg_df is None:
            invalid_patients += 1
            continue

        filtered_df = filter_ctg_data(
            ctg_df, min_data=min_data, sample_rate_hz=sample_rate_hz
        )
        if filtered_df is None:
            invalid_patients += 1
            continue

        baby_id = next_baby_id
        next_baby_id += 1
        valid_patients += 1

        filtered_df.insert(0, "BabyID", baby_id)
        _append_metadata(metadata_path, baby_id, birth_day, apgar5)

        table = pa.Table.from_pandas(
            filtered_df[["BabyID", "Timestamp", "FHR", "toco"]],
            preserve_index=False,
        )
        writer.write(table)

        if report_every and (valid_patients + invalid_patients) % report_every == 0:
            print(
                f"Processed {valid_patients + invalid_patients} patients "
                f"(valid={valid_patients}, invalid={invalid_patients})"
            )

    writer.close()

    print(
        f"Done. valid={valid_patients}, invalid={invalid_patients}, "
        f"batch_id={batch_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CTG preprocessing utilities.")
    parser.add_argument(
        "--patient-csv",
        type=str,
        default=DEFAULT_PATIENT_CSV,
        help="Path to patient CSV.",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        nargs="+",
        default=DEFAULT_CTGSOURCE_PATHS,
        help="One or more parquet files with CTG data.",
    )
    parser.add_argument(
        "--test-first-valid",
        action="store_true",
        help="Scan a small range and stop at first valid patient.",
    )
    parser.add_argument("--start", type=int, default=1, help="Start patient index.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of patients to scan in test mode.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=40,
        help="How many rows of FHR/toco to print.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot FHR and toco for the first valid patient found.",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="ctg_plot.png",
        help="Output path for plot image.",
    )
    parser.add_argument(
        "--valid-index",
        type=int,
        default=1,
        help="Which valid patient to select within the scan range (1-based).",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Run full processing to create metadata.csv and parquet output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metadata.csv and parquet output.",
    )
    parser.add_argument(
        "--batch-id",
        type=int,
        default=1,
        help="Batch number for the output parquet file.",
    )
    parser.add_argument(
        "--min-data",
        type=float,
        default=0.5,
        help="Minimum fraction of valid data required (0-1).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=DEFAULT_REPORT_EVERY,
        help="Print progress every N patients.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[1, 4],
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Sample rate in Hz for processing (1 or 4).",
    )
    parser.add_argument(
        "--downsample-mode",
        type=str,
        choices=["mean", "first"],
        default=DEFAULT_DOWNSAMPLE_MODE,
        help="How to reduce 4 samples/second to 1 Hz when sample-rate=1.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["append", "dataset"],
        default=DEFAULT_OUTPUT_MODE,
        help="Append to an existing parquet file or write a new dataset file.",
    )
    parser.add_argument(
        "--max-batch-size-gb",
        type=float,
        default=DEFAULT_MAX_BATCH_SIZE_GB,
        help="Auto-switch to dataset output when append file exceeds this size.",
    )

    args = parser.parse_args()

    if args.test_first_valid:
        print(
            f"Scanning patients {args.start}..{args.start + args.limit - 1} "
            f"for first valid CTG."
        )
        found = False
        valid_seen = 0
        dataset = _build_dataset(args.parquet)
        for patient_index, patient in iter_patients(
            args.patient_csv, args.start, args.limit
        ):

            apgar5 = patient.get("apgar5")
            if apgar5 is None:
                continue

            ctg_df = load_ctg_data(
                dataset,
                patient.get("pn", ""),
                patient.get("birth_day"),
                sample_rate_hz=args.sample_rate,
                downsample_mode=args.downsample_mode,
                bucket_count=DEFAULT_PARTITION_BUCKETS,
            )
            if ctg_df is None:
                continue

            filtered_df = filter_ctg_data(ctg_df, sample_rate_hz=args.sample_rate)
            if filtered_df is None:
                continue

            valid_seen += 1
            if valid_seen < args.valid_index:
                continue

            found = True
            print(f"Found valid patient at row {patient_index}.")
            print(f"Apgar5: {apgar5}")
            print(
                f"Rows: {len(filtered_df)}, "
                f"filtered ratio: {filtered_df['filtered'].mean():.2%}"
            )
            print(
                f"Timestamp range: {filtered_df['Timestamp'].min()} "
                f"to {filtered_df['Timestamp'].max()}"
            )
            fhr_nan = filtered_df["FHR"].isna().sum()
            print(f"FHR NaNs after filtering: {fhr_nan}")

            preview = filtered_df[["Timestamp", "FHR", "toco", "filtered"]]
            print("First rows:")
            print(preview.head(args.print_limit).to_string(index=False))
            print("Last rows:")
            print(preview.tail(args.print_limit).to_string(index=False))

            if args.plot:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
                axes[0].plot(filtered_df["Timestamp"], filtered_df["FHR"], linewidth=0.8)
                axes[0].set_ylabel("FHR")
                axes[0].set_title(f"CTG FHR ({args.sample_rate} Hz)")

                axes[1].plot(filtered_df["Timestamp"], filtered_df["toco"], linewidth=0.8)
                axes[1].set_ylabel("Toco")
                axes[1].set_title(f"Tocography ({args.sample_rate} Hz)")

                axes[1].set_xlabel("Timestamp")
                fig.tight_layout()
                plot_path = Path(args.plot_out)
                fig.savefig(plot_path, dpi=150)
                print(f"Saved plot to {plot_path}")
            break

        if not found:
            print("No valid patients found in this range.")
        return

    if args.process:
        process_patients(
            patient_csv=args.patient_csv,
            parquet_paths=args.parquet,
            output_dir=args.output_dir,
            start_patient=args.start,
            patient_count=args.limit,
            batch_id=args.batch_id,
            min_data=args.min_data,
            sample_rate_hz=args.sample_rate,
            downsample_mode=args.downsample_mode,
            report_every=args.report_every,
            output_mode=args.output_mode,
            max_batch_size_gb=args.max_batch_size_gb,
        )
        return

    print("Hello from ctg-preprocess! Use --test-first-valid or --process to run.")


if __name__ == "__main__":
    main()
