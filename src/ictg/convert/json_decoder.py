import glob
import io
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

import pandas as pd
from pydantic import ValidationError

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX, open_json_stream
from .logging_config import get_logger
from .pydanticModels import PatientRecord, normalize_patient_record

logger = get_logger()

FailureType = Literal["parse", "validation"]


@dataclass(frozen=True)
class ConversionFailure:
    source: str
    error_type: FailureType
    location: str
    message: str


def _single_line_message(message: str) -> str:
    return " ".join(str(message).split())


def format_failure_detail(failure: ConversionFailure) -> str:
    return (
        f"{failure.error_type} failure in {failure.source} "
        f"at {failure.location}: {failure.message}"
    )


def format_failure_summary(
    failure_count: int,
    grouped_failures: list[tuple[str, int]],
    top_n: int = 5,
) -> list[str]:
    summary = [f"Encountered {failure_count:,} conversion failures"]
    if grouped_failures:
        summary.append("Most problematic files:")
        for source, count in grouped_failures[:top_n]:
            summary.append(f"  {source}: {count:,} failures")
    return summary


class FailureTracker:
    def __init__(self) -> None:
        self._failures: list[ConversionFailure] = []

    def add(
        self,
        *,
        source: str,
        error_type: FailureType,
        location: str,
        message: str,
    ) -> None:
        self._failures.append(
            ConversionFailure(
                source=source,
                error_type=error_type,
                location=location,
                message=_single_line_message(message),
            )
        )

    @property
    def count(self) -> int:
        return len(self._failures)

    def failures(self) -> list[ConversionFailure]:
        return list(self._failures)

    def grouped_counts(self) -> list[tuple[str, int]]:
        return Counter(failure.source for failure in self._failures).most_common()


def _format_size(size_bytes: float) -> str:
    """Format bytes as MB or GB depending on size."""
    mb = size_bytes / (1024 * 1024)
    if mb >= 1000:
        gb = mb / 1024
        return f"{gb:.1f} GB"
    else:
        return f"{mb:.1f} MB"


class ProgressReporter:
    """Adaptive progress reporter for streaming operations."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.report_count = 0
        self.FREQUENT_REPORTING_LIMIT = 5

    def __call__(
        self,
        objects_yielded: int,
        bytes_read: int,
        total_bytes: Optional[int],
    ) -> None:
        self.report(
            objects_yielded=objects_yielded,
            bytes_read=bytes_read,
            total_bytes=total_bytes,
        )

    def report(
        self,
        objects_yielded: int,
        bytes_read: int,
        total_bytes: Optional[int],
    ) -> None:
        """Report progress with adaptive frequency."""
        if not self.__should_report(objects_yielded):
            return

        self.report_count += 1
        if total_bytes is not None:
            percent = (bytes_read / total_bytes) * 100
            if self.report_count <= self.FREQUENT_REPORTING_LIMIT:
                logger.info(
                    "Streamed %s of %s (%.1f%%), parsed %s JSON objects",
                    _format_size(bytes_read),
                    _format_size(total_bytes),
                    percent,
                    f"{objects_yielded:,}",
                )
            else:
                logger.info(
                    "Streamed %s of %s (%.1f%%) in %s, parsed %s JSON objects",
                    _format_size(bytes_read),
                    _format_size(total_bytes),
                    percent,
                    self.__elapsed_time(),
                    f"{objects_yielded:,}",
                )
        else:
            if self.report_count <= self.FREQUENT_REPORTING_LIMIT:
                logger.info(
                    "Streamed %s, parsed %s JSON objects",
                    _format_size(bytes_read),
                    f"{objects_yielded:,}",
                )
            else:
                logger.info(
                    "Streamed %s in %s, parsed %s JSON objects",
                    _format_size(bytes_read),
                    self.__elapsed_time(),
                    f"{objects_yielded:,}",
                )

        self.__warn_frequency_change()

    def __elapsed_time(self) -> str:
        """
        Get the elapsed time since the start of the reporting.

        Convert to hours/minutes/seconds as appropriate.
        """
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"

        if elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m {seconds:.1f}s"

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"

    def __should_report(self, objects_yielded: int) -> bool:
        """Determine if progress should be reported based on object count."""
        if self.report_count < self.FREQUENT_REPORTING_LIMIT:
            return objects_yielded % 100_000 == 0
        else:
            return objects_yielded % 1_000_000 == 0

    def __warn_frequency_change(self) -> None:
        """Warn about switching to less frequent updates."""
        if self.report_count == self.FREQUENT_REPORTING_LIMIT:
            logger.info("Switching to less frequent progress updates")


def iter_concatenated_json(
    stream: io.TextIOWrapper,
    source: str,
    failure_tracker: Optional[FailureTracker] = None,
    chunk_size: int = 1 << 20,
    total_bytes: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Incrementally parse a stream that contains many top-level JSON objects back-to-back,
    possibly separated by newlines, without wrapping [] and commas. Constant memory.
    """
    dec = json.JSONDecoder()
    buf = ""
    eof = False
    bytes_read = 0
    buffer_start_offset = 0
    objects_yielded = 0
    reporter = ProgressReporter()

    while True:
        while True:
            s = buf.lstrip()
            leading_whitespace = len(buf) - len(s)
            if leading_whitespace:
                buffer_start_offset += len(buf[:leading_whitespace].encode("utf-8"))
            if not s:
                buf = ""
                break
            try:
                obj, end = dec.raw_decode(s)
                if isinstance(obj, dict):
                    objects_yielded += 1
                    reporter(
                        objects_yielded=objects_yielded,
                        bytes_read=bytes_read,
                        total_bytes=total_bytes,
                    )
                    yield obj
                else:
                    logger.warning(
                        "Skipping non-dict JSON object of type %s",
                        type(obj).__name__,
                    )
                buf = s[end:]
                buffer_start_offset += len(s[:end].encode("utf-8"))
            except json.JSONDecodeError as exc:
                next_object_start = s.find("{", max(exc.pos, 1))
                if next_object_start == -1:
                    buf = s
                    break

                if failure_tracker is not None:
                    failure_tracker.add(
                        source=source,
                        error_type="parse",
                        location=(
                            f"byte offset "
                            f"{buffer_start_offset + len(s[: exc.pos].encode('utf-8'))}"
                        ),
                        message=exc.msg,
                    )

                skipped = s[:next_object_start]
                buffer_start_offset += len(skipped.encode("utf-8"))
                buf = s[next_object_start:]
            except ValueError:
                buf = s
                break

        if eof:
            if buf.strip():
                if failure_tracker is not None:
                    failure_tracker.add(
                        source=source,
                        error_type="parse",
                        location=f"byte offset {buffer_start_offset}",
                        message="Trailing incomplete/invalid JSON at end of stream.",
                    )
            if total_bytes is not None:
                logger.info(
                    "Finished streaming: %s processed, %s JSON objects parsed",
                    _format_size(total_bytes),
                    f"{objects_yielded:,}",
                )
            else:
                logger.info(
                    "Finished streaming: %s processed, %s JSON objects parsed",
                    _format_size(bytes_read),
                    f"{objects_yielded:,}",
                )
            return

        chunk = stream.read(chunk_size)
        bytes_read += len(chunk.encode("utf-8"))
        if chunk == "":
            eof = True
        else:
            buf += chunk


def records_from_path(
    path: Path,
    member: Optional[str] = None,
    failure_tracker: Optional[FailureTracker] = None,
) -> Iterator[PatientRecord]:
    # pylint: disable=contextmanager-generator-missing-cleanup
    total_bytes = None
    source = path.name
    if path.suffix.lower() == JSON_SUFFIX:
        total_bytes = path.stat().st_size
    elif path.suffix.lower() == ZIP_SUFFIX:
        import zipfile

        with zipfile.ZipFile(path) as zf:
            name = member
            if name is None:
                from .file_reader import _pick_zip_json_member

                name = _pick_zip_json_member(zf)
            zi = zf.getinfo(name)
            total_bytes = zi.file_size
            source = f"{path.name}::{name}"

    record_index = 0
    with open_json_stream(path, member) as fp:
        for obj in iter_concatenated_json(
            fp,
            source=source,
            failure_tracker=failure_tracker,
            total_bytes=total_bytes,
        ):
            record_index += 1
            try:
                yield PatientRecord.model_validate(obj)
            except ValidationError as e:
                if failure_tracker is not None:
                    failure_tracker.add(
                        source=source,
                        error_type="validation",
                        location=f"record index {record_index}",
                        message=str(e),
                    )
                continue


def dataframe_from_glob(
    glob_exprs: list[str],
    member: Optional[str] = None,
    limit: Optional[int] = None,
    failure_tracker: Optional[FailureTracker] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for glob_expr in glob_exprs:
        for path_str in sorted(glob.glob(glob_expr)):
            path = Path(path_str)
            if (
                path.suffix.lower() not in {JSON_SUFFIX, ZIP_SUFFIX}
                or not path.is_file()
            ):
                logger.warning("Skipping unsupported file: %s", path)
                continue

            logger.info("Processing file: %s", path.name)
            for rec in records_from_path(
                path, member, failure_tracker=failure_tracker
            ):
                rows.append(normalize_patient_record(rec))
                if len(rows) % 1000 == 0:
                    logger.info("Processed %s records so far...", f"{len(rows):,}")
                if limit is not None and len(rows) >= limit:
                    logger.info("Reached limit of %s records", f"{limit:,}")
                    df = pd.DataFrame(rows)
                    return df
    logger.info("Total records processed: %s", f"{len(rows):,}")
    df = pd.DataFrame(rows)
    return df
