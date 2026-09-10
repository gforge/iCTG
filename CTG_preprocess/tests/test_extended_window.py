"""Stage 8b: the last N minutes before the final-window end, cut from all sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from extended_window import build_extended_window


def _ts(values: list[str]) -> pa.Array:
    return pa.array([datetime.fromisoformat(v) for v in values], pa.timestamp("us"))


def test_extended_window_cuts_history_and_shifts(tmp_path: Path) -> None:
    all_dir = tmp_path / "all_sessions"
    all_dir.mkdir()
    # A: sessions at 08:00-08:02 and 10:00-11:00 (end 11:00); rows every 30 min for brevity
    times_a = [
        "2016-01-10 08:00",
        "2016-01-10 08:02",
        "2016-01-10 09:30",
        "2016-01-10 10:00",
        "2016-01-10 10:30",
        "2016-01-10 11:00",
    ]
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A"] * len(times_a) + ["Z"],
                "session_id": [1, 1, 2, 2, 2, 2, 1],
                "Timestamp": _ts(times_a + ["2016-01-10 10:00"]),
                "FHR": [120.0, 121.0, 130.0, 140.0, 141.0, 142.0, 150.0],
                "toco": [5.0] * 7,
                "Hr1_SignalQuality": ["Y"] * 7,
                "Hr1Mode": ["US"] * 7,
                "TocoMode": ["EXT"] * 7,
                "in_final_window": [False, False, False, True, True, True, False],
            }
        ),
        all_dir / "all_sessions_bucket_0000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A", "A"],
                "Timestamp": _ts(["2016-01-10 10:00", "2016-01-10 11:00"]),
                "FHR": [140.0, 142.0],
            }
        ),
        tmp_path / "ctg_final.parquet",
    )
    pq.write_table(pa.table({"BabyID": ["A", "Z"], "shift_days": [3, 9]}), tmp_path / "key.parquet")

    summary = build_extended_window(
        120,
        all_sessions_dir=all_dir,
        ctg_final=tmp_path / "ctg_final.parquet",
        key_file=tmp_path / "key.parquet",
        out=tmp_path / "ctg_final_120m.parquet",
    )
    assert summary == {"minutes": 120, "rows": 4, "babies": 1, "matched_babies": 1}
    out = pq.read_table(tmp_path / "ctg_final_120m.parquet").to_pandas().sort_values("Timestamp")
    assert list(out.columns) == [
        "BabyID",
        "Timestamp",
        "FHR",
        "toco",
        "Hr1_SignalQuality",
        "Hr1Mode",
        "TocoMode",
    ]
    # window (09:00, 11:00]: 09:30, 10:00, 10:30, 11:00 -> shifted by +3 days; Z is not matched
    assert out["Timestamp"].iloc[0] == pd.Timestamp("2016-01-13 09:30")
    assert out["Timestamp"].iloc[-1] == pd.Timestamp("2016-01-13 11:00")
    assert set(out["BabyID"]) == {"A"}
