import glob
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from pydantic import ValidationError

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX, open_json_stream
from .pydanticModels import PatientRecord, normalize_patient_record


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
                print(
                    f"[info] Streamed {_format_size(bytes_read)} of "
                    f"{_format_size(total_bytes)} ({percent:.1f}%)"
                    f", parsed {objects_yielded:,} JSON objects",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[info] Streamed {_format_size(bytes_read)} of "
                    f"{_format_size(total_bytes)} ({percent:.1f}%) in {self.__elapsed_time()}"
                    f", parsed {objects_yielded:,} JSON objects",
                    file=sys.stderr,
                )
        else:
            if self.report_count <= self.FREQUENT_REPORTING_LIMIT:
                print(
                    f"[info] Streamed {_format_size(bytes_read)}"
                    f", parsed {objects_yielded:,} JSON objects",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[info] Streamed {_format_size(bytes_read)} in {self.__elapsed_time()}"
                    f", parsed {objects_yielded:,} JSON objects",
                    file=sys.stderr,
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
            print(
                "[info] Switching to less frequent progress updates",
                file=sys.stderr,
            )


def iter_concatenated_json(
    stream: io.TextIOWrapper,
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
    objects_yielded = 0
    reporter = ProgressReporter()

    while True:
        while True:
            s = buf.lstrip()
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
                    print(
                        f"[warn] Skipping non-dict JSON object of type {type(obj).__name__}",
                        file=sys.stderr,
                    )
                buf = s[end:]
            except ValueError:
                buf = s
                break

        if eof:
            if buf.strip():
                raise ValueError("Trailing incomplete/invalid JSON at end of stream.")
            if total_bytes is not None:
                print(
                    f"[info] Finished streaming: {_format_size(total_bytes)} processed, "
                    f"{objects_yielded:,} JSON objects parsed",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[info] Finished streaming: {_format_size(bytes_read)} processed, "
                    f"{objects_yielded:,} JSON objects parsed",
                    file=sys.stderr,
                )
            return

        chunk = stream.read(chunk_size)
        bytes_read += len(chunk)
        if chunk == "":
            eof = True
        else:
            buf += chunk


def records_from_path(
    path: Path, member: Optional[str] = None
) -> Iterator[PatientRecord]:
    # pylint: disable=contextmanager-generator-missing-cleanup
    total_bytes = None
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

    with open_json_stream(path, member) as fp:
        for obj in iter_concatenated_json(fp, total_bytes=total_bytes):
            try:
                yield PatientRecord.model_validate(obj)
            except ValidationError as e:
                # Surface context but keep streaming
                print(f"[warn] Validation error in {path.name}: {e}", file=sys.stderr)
                continue


def dataframe_from_glob(
    glob_exprs: list[str], member: Optional[str] = None, limit: Optional[int] = None
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for glob_expr in glob_exprs:
        for path_str in sorted(glob.glob(glob_expr)):
            path = Path(path_str)
            if (
                path.suffix.lower() not in {JSON_SUFFIX, ZIP_SUFFIX}
                or not path.is_file()
            ):
                print(f"[warn] Skipping unsupported file: {path}", file=sys.stderr)
                continue

            print(f"[info] Processing file: {path.name}", file=sys.stderr)
            for rec in records_from_path(path, member):
                rows.append(normalize_patient_record(rec))
                if len(rows) % 1000 == 0:
                    print(
                        f"[info] Processed {len(rows):,} records so far...",
                        file=sys.stderr,
                    )
                if limit is not None and len(rows) >= limit:
                    print(f"[info] Reached limit of {limit:,} records", file=sys.stderr)
                    df = pd.DataFrame(rows)
                    return df
    print(f"[info] Total records processed: {len(rows):,}", file=sys.stderr)
    df = pd.DataFrame(rows)
    return df
