"""Write validated CTG records to restart-safe parquet outputs."""

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX
from .json_decoder import FailureTracker, records_from_path
from .logging_config import get_logger
from .pydanticModels import normalize_patient_record

logger = get_logger()
PARQUET_FIELDS: list[pa.Field] = [
    pa.field("PatientID", pa.string()),
    pa.field("RegistrationID", pa.int64()),
    pa.field("Timestamp", pa.timestamp("us")),
    pa.field("Hr1Mode", pa.string()),
    pa.field("Hr2Mode", pa.string()),
    pa.field("MhrMode", pa.string()),
    pa.field("TocoMode", pa.string()),
    pa.field("Toco_Values", pa.string()),
    pa.field("Hr1_0", pa.int16()),
    pa.field("Hr1_1", pa.int16()),
    pa.field("Hr1_2", pa.int16()),
    pa.field("Hr1_3", pa.int16()),
    pa.field("Hr1_SignalQuality", pa.string()),
    pa.field("Hr2_0", pa.int16()),
    pa.field("Hr2_1", pa.int16()),
    pa.field("Hr2_2", pa.int16()),
    pa.field("Hr2_3", pa.int16()),
    pa.field("Hr2_SignalQuality", pa.string()),
    pa.field("Mhr_0", pa.int16()),
    pa.field("Mhr_1", pa.int16()),
    pa.field("Mhr_2", pa.int16()),
    pa.field("Mhr_3", pa.int16()),
    pa.field("Mhr_SignalQuality", pa.string()),
]
PARQUET_SCHEMA = pa.schema(PARQUET_FIELDS)


def write_parquet_per_input(
    glob_exprs: list[str],
    out_dir: Path,
    member: Optional[str] = None,
    batch_size: int = 50_000,
    skip_existing: bool = False,
    failure_tracker: Optional[FailureTracker] = None,
) -> None:
    """Convert each matching input file into a parquet file in `out_dir`."""
    for glob_expr in glob_exprs:
        for path_str in sorted(glob.glob(glob_expr)):
            path = Path(path_str)
            output_paths = _prepare_output_paths(
                path,
                out_dir,
                skip_existing=skip_existing,
            )
            if output_paths is None:
                continue

            out_path = output_paths["final"]
            temp_path = output_paths["temp"]
            logger.info("Starting %s -> %s", path, out_path)

            writer: Optional[pq.ParquetWriter] = None
            batch: List[Dict[str, Any]] = []
            total_records = 0
            try:
                for rec in records_from_path(
                    path, member, failure_tracker=failure_tracker
                ):
                    batch.append(normalize_patient_record(rec))
                    if len(batch) >= batch_size:
                        _flush_parquet_batch(
                            batch, temp_path, writer_container := {"writer": writer}
                        )
                        writer = writer_container["writer"]
                        total_records += len(batch)
                        logger.info(
                            "Flushed %s records from %s (%s total written, batch size %s)",
                            f"{len(batch):,}",
                            path.name,
                            f"{total_records:,}",
                            f"{batch_size:,}",
                        )
                        batch.clear()
                if batch:
                    _flush_parquet_batch(
                        batch, temp_path, writer_container := {"writer": writer}
                    )
                    writer = writer_container["writer"]
                    total_records += len(batch)
                    logger.info(
                        "Flushed %s records from %s (%s total written, final batch)",
                        f"{len(batch):,}",
                        path.name,
                        f"{total_records:,}",
                    )
                if writer is not None:
                    writer.close()
                    writer = None
                if total_records == 0:
                    logger.warning(
                        "No valid records were written for %s; leaving final output unchanged",
                        path,
                    )
                    continue
                temp_path.replace(out_path)
                logger.info(
                    "Completed %s: wrote %s total records to %s",
                    path.name,
                    f"{total_records:,}",
                    out_path,
                )
            except Exception:
                logger.exception(
                    "Failed processing %s -> %s (temp file: %s)",
                    path,
                    out_path,
                    temp_path,
                )
                raise
            finally:
                if writer is not None:
                    writer.close()
                if batch:
                    batch.clear()


def _prepare_output_paths(
    path: Path,
    out_dir: Path,
    *,
    skip_existing: bool,
) -> Dict[str, Path] | None:
    if path.suffix.lower() not in {JSON_SUFFIX, ZIP_SUFFIX} or not path.is_file():
        logger.warning("Skipping unsupported file: %s", path)
        return None

    out_path = out_dir / (path.stem + ".parquet")
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    if out_path.exists() and skip_existing:
        if temp_path.exists():
            temp_path.unlink()
            logger.info(
                "Removed stale temporary file for skipped output: %s", temp_path
            )
        logger.info("Skipping existing output: %s", out_path)
        return None

    if temp_path.exists():
        temp_path.unlink()
        logger.info("Removed stale temporary file: %s", temp_path)

    if out_path.exists():
        logger.info("Existing output will be replaced on success: %s", out_path)

    return {"final": out_path, "temp": temp_path}


def _flush_parquet_batch(
    batch: List[Dict[str, Any]], out_path: Path, writer_container: Dict[str, Any]
) -> None:
    table = pa.Table.from_pylist(batch, schema=PARQUET_SCHEMA)
    writer = writer_container.get("writer")
    if writer is None:
        writer = pq.ParquetWriter(
            out_path.as_posix(),
            PARQUET_SCHEMA,
            compression="zstd",
        )
        writer_container["writer"] = writer
        logger.info("Created Parquet file: %s", out_path)
    writer.write_table(table)
    logger.debug("Flushed batch of %s records to %s", f"{len(batch):,}", out_path)
