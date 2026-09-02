"""Unit tests for the raw CTG JSON -> parquet converter."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ictg.convert.json_decoder import FailureTracker, iter_concatenated_json
from ictg.convert.pydanticModels import PatientRecord, normalize_patient_record
from ictg.convert.write_parquet import PARQUET_SCHEMA, write_parquet_per_input


def _record(patient: str = "199001011234", ts: str = "7/24/2021 1:16:02 PM") -> dict:
    return {
        "PatientID": patient,
        "RegistrationID": 42,
        "Timestamp": ts,
        "Hr1Mode": "US",
        "Hr1": {"Values": [140, 141, 0, 139], "SignalQuality": "Green"},
        "Hr2": None,
        "Mhr": {"Values": [80, 81]},
        "Toco": {"Values": "AQID"},
    }


def test_patient_record_parses_us_timestamp_and_flattens() -> None:
    rec = PatientRecord.model_validate(_record())
    assert rec.Timestamp == datetime(2021, 7, 24, 13, 16, 2)

    row = normalize_patient_record(rec)
    assert row["PatientID"] == "199001011234"
    assert row["Hr1_0"] == 140
    assert row["Hr1_3"] == 139
    assert row["Hr1_SignalQuality"] == "Green"
    # Hr2 missing -> all columns present but None
    assert row["Hr2_0"] is None
    assert row["Hr2_SignalQuality"] is None
    # Mhr has only two values -> remaining positions padded with None
    assert row["Mhr_1"] == 81
    assert row["Mhr_2"] is None
    assert row["Toco_Values"] == "AQID"
    assert set(row) == {f.name for f in PARQUET_SCHEMA}


def test_iter_concatenated_json_handles_back_to_back_objects_and_garbage() -> None:
    good1 = json.dumps(_record())
    good2 = json.dumps(_record(patient="199202022345"))
    stream = io.StringIO(f"{good1}\n{{not json}}{good2}   ")
    tracker = FailureTracker()

    objects = list(
        iter_concatenated_json(stream, source="test", failure_tracker=tracker, chunk_size=16)
    )

    assert [o["PatientID"] for o in objects] == ["199001011234", "199202022345"]
    assert tracker.count == 1
    (failure,) = tracker.failures()
    assert failure.error_type == "parse"


def test_write_parquet_per_input_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "Export_test.json"
    src.write_text(
        "\n".join(
            [
                json.dumps(_record()),
                json.dumps(_record(ts="not a timestamp")),  # validation failure
                json.dumps(_record(patient="199202022345")),
            ]
        )
    )
    out_dir = tmp_path / "out"
    tracker = FailureTracker()

    write_parquet_per_input([str(src)], out_dir, batch_size=1, failure_tracker=tracker)

    out_file = out_dir / "Export_test.parquet"
    assert out_file.exists()
    assert not (out_dir / "Export_test.parquet.tmp").exists()
    table = pq.read_table(out_file)
    assert table.num_rows == 2
    assert table.schema.equals(PARQUET_SCHEMA)
    assert tracker.count == 1
    assert tracker.failures()[0].error_type == "validation"


def test_write_parquet_skip_existing_leaves_file_untouched(tmp_path: Path) -> None:
    src = tmp_path / "Export_test.json"
    src.write_text(json.dumps(_record()))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "Export_test.parquet"
    existing.write_bytes(b"sentinel")

    write_parquet_per_input([str(src)], out_dir, skip_existing=True)

    assert existing.read_bytes() == b"sentinel"


@pytest.mark.parametrize("bad", ["2021-07-24 13:16:02", 12345])
def test_patient_record_rejects_unknown_timestamp_formats(bad: object) -> None:
    with pytest.raises(ValueError):
        PatientRecord.model_validate(_record(ts=bad))  # type: ignore[arg-type]
