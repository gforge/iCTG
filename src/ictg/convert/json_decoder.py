import glob
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from pydantic import ValidationError

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX, open_json_stream
from .pydanticModels import PatientRecord, normalize_patient_record


def iter_concatenated_json(
    stream: io.TextIOWrapper, chunk_size: int = 1 << 20
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
                    if objects_yielded % 1e5 == 0:
                        mb_read = bytes_read / (1024 * 1024)
                        print(
                            f"[info] Streamed {mb_read:.1f} MB, "
                            f"parsed {objects_yielded:,} JSON objects",
                            file=sys.stderr,
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
            mb_read = bytes_read / (1024 * 1024)
            print(
                f"[info] Finished streaming: {mb_read:.1f} MB processed, "
                f"{objects_yielded:,} JSON objects parsed",
                file=sys.stderr,
            )
            return

        chunk = stream.read(chunk_size)
        bytes_read += len(chunk)
        # Print progress every 10 MB
        mb_read = bytes_read / (1024 * 1024)
        if chunk == "":
            eof = True
        else:
            buf += chunk


def records_from_path(
    path: Path, member: Optional[str] = None
) -> Iterator[PatientRecord]:
    # pylint: disable=contextmanager-generator-missing-cleanup
    with open_json_stream(path, member) as fp:
        for obj in iter_concatenated_json(fp):
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
