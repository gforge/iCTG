"""Convert the Milou ``ExportSignatures_*.json`` event exports to parquet.

Each file holds one JSON array (or concatenated objects) of events tied to a CTG
registration by ``RegistrationID``. Event types seen in the exports:

* ``Signature Event``  clinician's CTG classification (baseline, variability, accelerations,
                       decelerations, stage, status, twin) and the signing user;
* ``Mspo2Event``       maternal pulse oximetry (Hr, Mspo2, HrInvalid);
* ``NibpEvent``        maternal blood pressure (Systolic, Diastolic, Mean, HR);
* ``Lactate Event``    fetal scalp lactate (Lactate, MedicalTime);
* ``pH Event``         fetal scalp pH (PH);
* ``UserNoteEvent``    free-text note (NoteText).

``UserName`` (staff identity) is dropped. Unknown keys are ignored, malformed segments are
skipped and counted. Values are typed leniently: anything that fails to parse becomes NULL.
"""

from __future__ import annotations

import glob
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .logging_config import get_logger

logger = get_logger()

TIME_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")

EVENT_FIELDS: list[pa.Field] = [
    pa.field("RegistrationID", pa.int64()),
    pa.field("EventID", pa.int64()),
    pa.field("EventType", pa.string()),
    pa.field("Time", pa.timestamp("us")),
    pa.field("MedicalTime", pa.timestamp("us")),
    pa.field("BaseLine", pa.string()),
    pa.field("Variability", pa.string()),
    pa.field("Acceleration", pa.string()),
    pa.field("Decelerations", pa.string()),
    pa.field("Stage", pa.string()),
    pa.field("Status", pa.string()),
    pa.field("Twin", pa.string()),
    pa.field("Hr", pa.int32()),
    pa.field("Mspo2", pa.int32()),
    pa.field("HrInvalid", pa.bool_()),
    pa.field("Systolic", pa.int32()),
    pa.field("Diastolic", pa.int32()),
    pa.field("Mean", pa.int32()),
    pa.field("NibpHR", pa.int32()),
    pa.field("Lactate", pa.float64()),
    pa.field("PH", pa.float64()),
    pa.field("NoteText", pa.string()),
]
EVENT_SCHEMA = pa.schema(EVENT_FIELDS)
# raw key -> schema column (keys not listed are dropped, e.g. UserName)
KEY_MAP = {name: name for name in EVENT_SCHEMA.names}
KEY_MAP["HR"] = "NibpHR"
DROPPED_KEYS = {"UserName"}


def parse_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "ja"}:
        return True
    if text in {"false", "0", "no", "nej"}:
        return False
    return None


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_CONVERTERS = {
    pa.int64(): _to_int,
    pa.int32(): _to_int,
    pa.float64(): _to_float,
    pa.bool_(): _to_bool,
    pa.string(): _to_text,
}


def normalize_event(obj: dict[str, Any]) -> dict[str, Any]:
    """Map one raw event object onto the parquet schema (typed, unknown keys dropped)."""
    row: dict[str, Any] = {name: None for name in EVENT_SCHEMA.names}
    for key, value in obj.items():
        column = KEY_MAP.get(key)
        if column is None:
            continue
        field_type = EVENT_SCHEMA.field(column).type
        if pa.types.is_timestamp(field_type):
            row[column] = parse_time(value)
        else:
            row[column] = _CONVERTERS[field_type](value)
    return row


def iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield every top-level dict in ``text`` whether it is a JSON array, concatenated objects
    or a damaged mix; unparseable segments are skipped."""
    decoder = json.JSONDecoder()
    pos = 0
    skipped = 0
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        except json.JSONDecodeError:
            pass  # fall back to scanning
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            skipped += 1
            pos = start + 1
            continue
        pos = end
        if isinstance(obj, dict):
            yield obj
    if skipped:
        logger.warning("Skipped %s unparseable JSON segments", f"{skipped:,}")


def convert_signature_file(path: Path, out_dir: Path) -> tuple[int, Path]:
    """Convert one export file; returns (events written, parquet path)."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = [normalize_event(obj) for obj in iter_json_objects(text)]
    table = pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".parquet")
    pq.write_table(table, out_path, compression="zstd")
    return table.num_rows, out_path


def convert_signature_files(
    inputs: list[str], out_dir: Path, skip_existing: bool = False
) -> dict[str, int]:
    """Convert every matching export file; returns events written per file."""
    written: dict[str, int] = {}
    paths: list[Path] = []
    for pattern in inputs:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No input files match {inputs}")
    for path in paths:
        target = out_dir / (path.stem + ".parquet")
        if skip_existing and target.exists():
            logger.info("Skipping %s (parquet exists)", path.name)
            continue
        logger.info("Converting %s", path.name)
        n, out_path = convert_signature_file(path, out_dir)
        logger.info("Wrote %s events to %s", f"{n:,}", out_path.name)
        written[path.name] = n
    return written


def main() -> None:
    import argparse

    from .logging_config import setup_logging

    ap = argparse.ArgumentParser(
        description="Convert ExportSignatures JSON event files to parquet."
    )
    ap.add_argument(
        "inputs", nargs="+", help="Input files or globs (e.g. 'ExportSignatures_*.json')"
    )
    ap.add_argument("--parquet-out", type=Path, required=True, help="Output directory")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()
    setup_logging(log_dir=args.log_dir or args.parquet_out)
    written = convert_signature_files(
        args.inputs, args.parquet_out, skip_existing=args.skip_existing
    )
    logger.info("Done: %s events in %s files", f"{sum(written.values()):,}", len(written))


if __name__ == "__main__":
    main()
