"""Signature/event export conversion on synthetic JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ictg.convert.signatures import (
    EVENT_SCHEMA,
    convert_signature_files,
    iter_json_objects,
    normalize_event,
    parse_time,
)

EVENTS = [
    {
        "EventID": 1,
        "EventType": "Signature Event",
        "RegistrationID": 10,
        "Time": "7/24/2021 1:16:02 PM",
        "BaseLine": "110 - 160 spm",
        "Variability": "5 - 25 spm",
        "Acceleration": "",
        "Decelerations": "Inga",
        "Stage": "DeliveryStage",
        "Status": "Normal",
        "Twin": "",
        "UserName": "should be dropped",
    },
    {
        "EventID": 2,
        "EventType": "Mspo2Event",
        "RegistrationID": 10,
        "Time": "7/24/2021 1:17:02 PM",
        "Hr": 88,
        "Mspo2": 97,
        "HrInvalid": False,
    },
    {
        "EventID": 3,
        "EventType": "NibpEvent",
        "RegistrationID": 10,
        "Time": "7/24/2021 1:18:02 PM",
        "Systolic": 120,
        "Diastolic": 80,
        "Mean": 93,
        "HR": 85,
    },
    {
        "EventID": 4,
        "EventType": "Lactate Event",
        "RegistrationID": 11,
        "MedicalTime": "7/24/2021 2:00:00 PM",
        "Lactate": "4,2",
    },
    {
        "EventID": 5,
        "EventType": "UserNoteEvent",
        "RegistrationID": 11,
        "Time": "7/24/2021 2:05:00 PM",
        "NoteText": "Bricanyl 0,25 mg sc",
    },
]


def test_parse_time_formats() -> None:
    assert parse_time("7/24/2021 1:16:02 PM") == datetime(2021, 7, 24, 13, 16, 2)
    assert parse_time("2021-07-24T13:16:02") == datetime(2021, 7, 24, 13, 16, 2)
    assert parse_time("") is None and parse_time("garbage") is None


def test_normalize_event_types_and_drops_username() -> None:
    row = normalize_event(EVENTS[0])
    assert set(row) == set(EVENT_SCHEMA.names)
    assert row["Acceleration"] is None and row["Twin"] is None  # empty strings -> NULL
    assert row["Status"] == "Normal" and "UserName" not in row
    assert normalize_event(EVENTS[2])["NibpHR"] == 85
    assert normalize_event(EVENTS[3])["Lactate"] == 4.2  # decimal comma
    assert normalize_event(EVENTS[1])["HrInvalid"] is False


def test_iter_json_objects_handles_array_concatenation_and_damage() -> None:
    as_array = json.dumps(EVENTS)
    assert len(list(iter_json_objects(as_array))) == 5
    concatenated = "\n".join(json.dumps(e) for e in EVENTS)
    assert len(list(iter_json_objects(concatenated))) == 5
    damaged = as_array[:-40] + "\n" + json.dumps(EVENTS[0])  # truncated array + one good object
    kinds = [o["EventID"] for o in iter_json_objects(damaged)]
    assert 1 in kinds and len(kinds) >= 4


def test_convert_signature_files_writes_parquet(tmp_path: Path) -> None:
    src = tmp_path / "ExportSignatures_test.json"
    src.write_text(json.dumps(EVENTS), encoding="utf-8")
    written = convert_signature_files([str(tmp_path / "ExportSignatures_*.json")], tmp_path / "out")
    assert written == {"ExportSignatures_test.json": 5}
    table = pq.read_table(tmp_path / "out" / "ExportSignatures_test.parquet")
    assert table.schema.equals(EVENT_SCHEMA)
    df = table.to_pandas().set_index("EventID")
    assert df.loc[4, "MedicalTime"] == datetime(2021, 7, 24, 14, 0, 0)
    assert str(df.loc[5, "NoteText"]).startswith("Bricanyl")
    assert df["EventType"].tolist() == [e["EventType"] for e in EVENTS]
