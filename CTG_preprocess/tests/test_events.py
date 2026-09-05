"""Stage 9 on synthetic data: registration map, pregnancy assignment, note flags, deliverable."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from events import run_stage9


def _ts_array(values: list[str | None]) -> pa.Array:
    return pa.array(
        [datetime.fromisoformat(v) if v is not None else None for v in values], pa.timestamp("us")
    )


@pytest.fixture
def stage9_inputs(tmp_path: Path) -> dict[str, Path]:
    ts = pd.to_datetime
    stage0 = tmp_path / "stage0"
    stage0.mkdir()
    # mother m1: registration 10 (2016 antenatal), 11 (2016 labour); mother m2: registration 20
    pq.write_table(
        pa.table(
            {
                "PatientID": ["m1", "m1", "m1", "m1", "m2", "m2"],
                "RegistrationID": [10, 10, 11, 11, 20, 20],
                "Timestamp": pa.array(
                    ts(
                        [
                            "2015-12-01 09:00",
                            "2015-12-01 09:30",
                            "2016-01-10 08:00",
                            "2016-01-10 11:00",
                            "2017-05-05 10:00",
                            "2017-05-05 12:00",
                        ]
                    )
                ),
                "Hr1_0": [140, 141, 142, 143, 130, 131],
            }
        ),
        stage0 / "export.parquet",
    )
    stage3 = tmp_path / "stage3"
    stage3.mkdir()
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A", "A", "C"],
                "PatientID": ["m1", "m1", "m2"],
                "Timestamp": pa.array(
                    ts(["2016-01-10 10:00", "2016-01-10 11:00", "2017-05-05 12:00"])
                ),
                "FHR": [140.0, 141.0, 130.0],
            }
        ),
        stage3 / "stage3_sessions_bucket_0000.parquet",
    )
    all_dir = stage3 / "all_sessions"
    all_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A", "A", "C"],
                "session_id": [1, 2, 1],
                "Timestamp": pa.array(
                    ts(["2015-12-01 09:00", "2016-01-10 10:00", "2017-05-05 12:00"])
                ),
                "FHR": [120.0, 140.0, 130.0],
            }
        ),
        all_dir / "all_sessions_bucket_0000.parquet",
    )
    events = tmp_path / "events"
    events.mkdir()
    pq.write_table(
        pa.table(
            {
                "RegistrationID": [10, 11, 11, 20, 99],
                "EventID": [1, 2, 3, 4, 5],
                "EventType": [
                    "Signature Event",
                    "UserNoteEvent",
                    "Lactate Event",
                    "NibpEvent",
                    "UserNoteEvent",
                ],
                "Time": _ts_array(
                    [
                        "2015-12-01 09:10",
                        "2016-01-10 09:00",
                        None,
                        "2017-05-05 11:00",
                        "2017-05-05 11:00",
                    ]
                ),
                "MedicalTime": _ts_array([None, None, "2016-01-10 10:30", None, None]),
                "BaseLine": ["110 - 160 spm", None, None, None, None],
                "Variability": [None] * 5,
                "Acceleration": [None] * 5,
                "Decelerations": ["Inga", None, None, None, None],
                "Stage": ["Fr o m 34+0", None, None, None, None],
                "Status": ["Normal", None, None, None, None],
                "Twin": [None] * 5,
                "Hr": pa.array([None] * 5, pa.int32()),
                "Mspo2": pa.array([None] * 5, pa.int32()),
                "HrInvalid": pa.array([None] * 5, pa.bool_()),
                "Systolic": pa.array([None, None, None, 120, None], pa.int32()),
                "Diastolic": pa.array([None, None, None, 80, None], pa.int32()),
                "Mean": pa.array([None, None, None, 93, None], pa.int32()),
                "NibpHR": pa.array([None, None, None, 85, None], pa.int32()),
                "Lactate": pa.array([None, None, 4.2, None, None], pa.float64()),
                "PH": pa.array([None] * 5, pa.float64()),
                "NoteText": [None, "Bricanyl 0,25 mg sc, EDA anlagd", None, None, "orphan note"],
            }
        ),
        events / "ExportSignatures_test.parquet",
    )
    pq.write_table(
        pa.table({"BabyID": ["A", "C"], "shift_days": [10, -5]}), tmp_path / "key.parquet"
    )
    return {"tmp": tmp_path, "stage0": stage0, "stage3": stage3, "all": all_dir, "events": events}


def test_stage9_links_events_and_writes_deliverable(stage9_inputs: dict[str, Path]) -> None:
    t = stage9_inputs["tmp"]
    summary = run_stage9(
        events_dir=stage9_inputs["events"],
        stage0_dir=stage9_inputs["stage0"],
        stage3_dir=stage9_inputs["stage3"],
        all_sessions_dir=stage9_inputs["all"],
        registration_map=t / "s9" / "registration_map.parquet",
        linked_out=t / "s9" / "events_linked.parquet",
        key_file=t / "key.parquet",
        shifted_out=t / "s8" / "events.parquet",
    )
    assert summary["registrations"] == 3
    assert summary["events_total"] == 5
    assert summary["events_without_known_registration"] == 1  # registration 99
    assert summary["events_linked"] == 4 and summary["pregnancies_with_events"] == 2

    linked = pq.read_table(t / "s9" / "events_linked.parquet").to_pandas().sort_values("Timestamp")
    assert linked["BabyID"].tolist() == ["A", "A", "A", "C"]  # antenatal registration 10 -> A too
    note = linked[linked["event_type"] == "UserNoteEvent"].iloc[0]
    assert (
        bool(note["note_bricanyl"])
        and bool(note["note_epidural"])
        and not bool(note["note_oxytocin"])
    )
    assert bool(note["note_present"]) and note["note_length"] > 10
    lactate = linked[linked["event_type"] == "Lactate Event"].iloc[0]
    assert lactate["scalp_lactate"] == 4.2 and lactate["Timestamp"] == pd.Timestamp(
        "2016-01-10 10:30"
    )
    assert linked[linked["BabyID"] == "C"]["bp_systolic"].iloc[0] == 120

    shifted = pq.read_table(t / "s8" / "events.parquet").to_pandas()
    assert "NoteText" not in shifted.columns and "UserName" not in shifted.columns
    assert "PatientID" not in shifted.columns and "RegistrationID" not in shifted.columns
    assert len(shifted) == 4
    a = shifted[shifted["BabyID"] == "A"].sort_values("Timestamp")
    assert a["Timestamp"].iloc[0] == pd.Timestamp("2015-12-01 09:10") + pd.Timedelta(days=10)
    c = shifted[shifted["BabyID"] == "C"].iloc[0]
    assert c["Timestamp"] == pd.Timestamp("2017-05-05 11:00") - pd.Timedelta(days=5)

    # the registration map is cached and reused
    assert (t / "s9" / "registration_map.parquet").exists()
    again = run_stage9(
        events_dir=stage9_inputs["events"],
        stage0_dir=t / "does-not-exist",
        stage3_dir=stage9_inputs["stage3"],
        all_sessions_dir=stage9_inputs["all"],
        registration_map=t / "s9" / "registration_map.parquet",
        linked_out=t / "s9" / "events_linked.parquet",
        key_file=t / "key.parquet",
        shifted_out=None,
    )
    assert again["events_linked"] == 4
