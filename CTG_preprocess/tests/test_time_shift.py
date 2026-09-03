"""Stage 8 time shifting on synthetic data: deterministic per-mother shifts, preserved birth
order with jittered intervals, and consistent application to registry, CTG and pretraining
outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from time_shift import build_shift_table, time_shift_outputs


def _ts(value: object) -> pd.Timestamp:
    return pd.Timestamp(str(value))


def _pregnancies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BabyID": ["m1_a", "m1_b", "m1_c", "m2_a", "orphan"],
            "PatientID": ["mother1", "mother1", "mother1", "mother2", None],
            "anchor": ["2016-01-10", "2018-01-10", "2018-07-10", "2017-05-05", "2019-09-09"],
        }
    )


def test_shift_table_is_deterministic_and_bounded() -> None:
    a = build_shift_table(_pregnancies(), "secret", max_days=100)
    b = build_shift_table(_pregnancies(), "secret", max_days=100)
    pd.testing.assert_frame_equal(a, b)
    c = build_shift_table(_pregnancies(), "other-secret", max_days=100)
    assert not a["shift_days"].equals(c["shift_days"])
    # the first pregnancy of every mother (and the orphan) is within the base range
    firsts = a.set_index("BabyID").loc[["m1_a", "m2_a", "orphan"], "shift_days"]
    assert (firsts.abs() <= 100).all()


def test_sibling_intervals_are_jittered_by_10_to_20_percent() -> None:
    preg = _pregnancies()
    table = build_shift_table(preg, "secret", max_days=365, jitter_min=0.10, jitter_max=0.20)
    merged = preg.merge(table, on="BabyID")
    merged["anchor"] = pd.to_datetime(merged["anchor"])
    merged["shifted"] = merged["anchor"] + pd.to_timedelta(merged["shift_days"], unit="D")
    m1 = merged[merged["PatientID"] == "mother1"].sort_values("anchor")
    true_gaps = m1["anchor"].diff().dt.days.dropna().to_numpy()
    shifted_gaps = m1["shifted"].diff().dt.days.dropna().to_numpy()
    assert (shifted_gaps > 0).all()  # order preserved
    ratio = shifted_gaps / true_gaps
    assert np.all((np.abs(ratio - 1) >= 0.10 - 1e-3) & (np.abs(ratio - 1) <= 0.20 + 1e-3))
    # siblings do not share one shift: the true spacing is not recoverable
    assert m1["shift_days"].nunique() > 1


def test_many_mothers_run_fast_and_use_full_range() -> None:
    n = 20000
    preg = pd.DataFrame(
        {
            "BabyID": [f"b{i}" for i in range(n)],
            "PatientID": [f"p{i // 2}" for i in range(n)],
            "anchor": pd.date_range("2015-01-01", periods=n, freq="D"),
        }
    )
    table = build_shift_table(preg, "s", max_days=365)
    firsts = table[table["BabyID"].str[1:].astype(int) % 2 == 0]["shift_days"]
    assert firsts.min() < -300 and firsts.max() > 300
    assert abs(float(firsts.mean())) < 20


def test_rejects_duplicate_babyids() -> None:
    preg = _pregnancies()
    preg.loc[1, "BabyID"] = "m1_a"
    with pytest.raises(ValueError):
        build_shift_table(preg, "s")


@pytest.fixture
def synthetic_stage_outputs(tmp_path: Path) -> dict[str, Path]:
    ts = pd.to_datetime
    stage3 = pa.table(
        {
            "BabyID": ["A", "A", "B", "C"],
            "PatientID": ["m1", "m1", "m1", "m2"],
            "Timestamp": pa.array(
                ts(["2016-01-10 10:00", "2016-01-10 11:00", "2018-01-10 09:00", "2017-05-05 12:00"])
            ),
            "FHR": [140.0, 141.0, 150.0, 130.0],
        }
    )
    pq.write_table(stage3, tmp_path / "stage3_sessions.parquet")
    all_dir = tmp_path / "all_sessions"
    all_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A", "A", "D"],
                "session_id": [1, 2, 1],
                "Timestamp": pa.array(
                    ts(["2015-11-01 08:00", "2016-01-10 10:00", "2019-09-09 01:00"])
                ),
                "FHR": [120.0, 140.0, 135.0],
                "in_final_window": [False, True, False],
            }
        ),
        all_dir / "all_sessions_bucket_0000.parquet",
    )
    registry = pd.DataFrame(
        {
            "BabyID": ["A", "B", "C"],
            "birth_day": ["2016-01-10", "2018-01-10", "2017-05-05"],
            "birth_timestamp": [
                "2016-01-10 11:30:00",
                "2018-01-10 09:45:00",
                "2017-05-05 12:30:00",
            ],
            "birth_time_seconds": [41400, 35100, 45000],
            "etablerade_varkar_datum": ["2016-01-10", None, "2017-05-04"],
            "etablerade_varkar_timestamp": ["2016-01-10 05:30:00", None, "2017-05-04 23:00:00"],
            "etablerade_varkar_seconds": [21600, None, 48600],
            "avled_datum": [None, "2018-01-12", None],
            "apgar5": [9, 4, 8],
        }
    )
    registry.to_csv(tmp_path / "registry.csv", index=False)
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A", "B", "C"],
                "Timestamp": pa.array(
                    ts(["2016-01-10 10:30", "2018-01-10 08:30", "2017-05-05 11:30"])
                ),
                "FHR": [140.0, 150.0, 130.0],
            }
        ),
        tmp_path / "ctg_final.parquet",
    )
    out = tmp_path / "stage8"
    summary = time_shift_outputs(
        stage3_path=tmp_path / "stage3_sessions.parquet",
        all_sessions_in=all_dir,
        registry_in=tmp_path / "registry.csv",
        ctg_in=tmp_path / "ctg_final.parquet",
        registry_out=out / "registry.csv",
        ctg_out=out / "ctg_final.parquet",
        all_sessions_out=out / "all_sessions",
        key_out=out / "timeshift_key.parquet",
        mothers_out=out / "mothers.csv",
        secret="fixture-secret",
        max_days=200,
    )
    assert summary["pregnancies"] == 4 and summary["babyids_without_patientid"] == 1
    return {"in": tmp_path, "out": out}


def test_outputs_are_shifted_consistently(synthetic_stage_outputs: dict[str, Path]) -> None:
    inp, out = synthetic_stage_outputs["in"], synthetic_stage_outputs["out"]
    key = pq.read_table(out / "timeshift_key.parquet").to_pandas().set_index("BabyID")["shift_days"]
    assert set(key.index) == {"A", "B", "C", "D"}
    assert (key != 0).all()

    reg_in = pd.read_csv(inp / "registry.csv").set_index("BabyID")
    reg_out = pd.read_csv(out / "registry.csv").set_index("BabyID")
    assert list(reg_out.columns) == list(reg_in.columns)
    for baby in ["A", "B", "C"]:
        d = pd.Timedelta(days=int(key[baby]))
        assert _ts(reg_out.loc[baby, "birth_day"]) == _ts(reg_in.loc[baby, "birth_day"]) + d
        assert (
            _ts(reg_out.loc[baby, "birth_timestamp"])
            == _ts(reg_in.loc[baby, "birth_timestamp"]) + d
        )
        # time of day and relative quantities untouched
        assert reg_out.loc[baby, "birth_time_seconds"] == reg_in.loc[baby, "birth_time_seconds"]
        assert reg_out.loc[baby, "apgar5"] == reg_in.loc[baby, "apgar5"]
    assert _ts(reg_out.loc["B", "avled_datum"]) == pd.Timestamp("2018-01-12") + pd.Timedelta(
        days=int(key["B"])
    )
    assert pd.isna(reg_out.loc["B", "etablerade_varkar_datum"])
    assert (
        reg_out["etablerade_varkar_seconds"].dropna()
        == reg_in["etablerade_varkar_seconds"].dropna()
    ).all()

    ctg_in = pq.read_table(inp / "ctg_final.parquet").to_pandas().set_index("BabyID")
    ctg_out = pq.read_table(out / "ctg_final.parquet").to_pandas().set_index("BabyID")
    for baby in ["A", "B", "C"]:
        assert ctg_out.loc[baby, "Timestamp"] == ctg_in.loc[baby, "Timestamp"] + pd.Timedelta(
            days=int(key[baby])
        )
    # the CTG of A ends on A's shifted birth day: registry and signal stay aligned
    assert ctg_out.loc["A", "Timestamp"].normalize() == _ts(reg_out.loc["A", "birth_day"])

    all_out = pq.read_table(out / "all_sessions" / "all_sessions_bucket_0000.parquet").to_pandas()
    all_in = pq.read_table(inp / "all_sessions" / "all_sessions_bucket_0000.parquet").to_pandas()
    assert len(all_out) == len(all_in)
    a_rows = all_out[all_out["BabyID"] == "A"].sort_values("session_id")
    a_in = all_in[all_in["BabyID"] == "A"].sort_values("session_id")
    assert (
        a_rows["Timestamp"].to_numpy()
        == (a_in["Timestamp"] + pd.Timedelta(days=int(key["A"]))).to_numpy()
    ).all()
    d_row = all_out[all_out["BabyID"] == "D"].iloc[0]
    assert d_row["Timestamp"] == pd.Timestamp("2019-09-09 01:00") + pd.Timedelta(days=int(key["D"]))


def test_missing_shift_is_an_error(tmp_path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "BabyID": ["A"],
                "PatientID": ["m"],
                "Timestamp": pa.array(pd.to_datetime(["2016-01-10"])),
            }
        ),
        tmp_path / "s3.parquet",
    )
    pd.DataFrame({"BabyID": ["A", "Z"], "birth_day": ["2016-01-10", "2016-02-02"]}).to_csv(
        tmp_path / "registry.csv", index=False
    )
    pq.write_table(
        pa.table({"BabyID": ["A"], "Timestamp": pa.array(pd.to_datetime(["2016-01-10"]))}),
        tmp_path / "ctg.parquet",
    )
    with pytest.raises(RuntimeError, match="registry"):
        time_shift_outputs(
            stage3_path=tmp_path / "s3.parquet",
            all_sessions_in=None,
            registry_in=tmp_path / "registry.csv",
            ctg_in=tmp_path / "ctg.parquet",
            registry_out=tmp_path / "out" / "registry.csv",
            ctg_out=tmp_path / "out" / "ctg.parquet",
            all_sessions_out=tmp_path / "out" / "all",
            key_out=tmp_path / "out" / "key.parquet",
            secret="s",
        )


def test_mothers_table_covers_every_pregnancy(synthetic_stage_outputs: dict[str, Path]) -> None:
    from pseudonyms import mother_id

    out = synthetic_stage_outputs["out"]
    mothers = pd.read_csv(out / "mothers.csv", dtype=str).fillna("")
    assert list(mothers.columns) == ["BabyID", "MotherID"]
    by_baby = mothers.set_index("BabyID")["MotherID"]
    assert set(by_baby.index) == {"A", "B", "C", "D"}
    assert by_baby["A"] == by_baby["B"] != by_baby["C"]  # A and B share mother m1
    assert by_baby["D"] == ""  # pretraining-only pregnancy without PatientID
    assert by_baby["A"] == mother_id("unit-test-salt", "m1")
