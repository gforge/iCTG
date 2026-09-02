"""Unit tests for the pure/small building blocks of ``ctg_reduction.py``."""

from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ctg_reduction import (
    _build_stage2_shards,
    _compute_fhr,
    _compute_toco,
    _stage3_babyid_expr,
    _stage3_bucket_expr,
)

# --------------------------------------------------------------------------------------
# Stage 2: shard planning over parquet row groups
# --------------------------------------------------------------------------------------


def _write_parquet_with_row_groups(path: Path, num_row_groups: int) -> None:
    table = pa.table({"x": list(range(num_row_groups))})
    pq.write_table(table, path, row_group_size=1)
    assert pq.ParquetFile(path).metadata.num_row_groups == num_row_groups


@pytest.mark.parametrize("shard_count", [1, 2, 3, 5, 8])
def test_build_stage2_shards_partitions_all_row_groups(tmp_path: Path, shard_count: int) -> None:
    row_groups_per_file = [3, 1, 4]
    paths = []
    for i, n in enumerate(row_groups_per_file):
        path = tmp_path / f"in-{i}.parquet"
        _write_parquet_with_row_groups(path, n)
        paths.append(path)

    plans, total = _build_stage2_shards(paths, shard_count)

    assert total == sum(row_groups_per_file) == 8
    assert len(plans) == shard_count

    # Every (file, row_group) is assigned exactly once, and each slice is non-empty & in range.
    assigned: list[tuple[Path, int]] = []
    for plan in plans:
        assert plan, "each shard must receive at least one row group"
        for path, start, stop in plan:
            assert 0 <= start < stop <= row_groups_per_file[paths.index(path)]
            assigned.extend((path, rg) for rg in range(start, stop))
    expected = {(p, rg) for p, n in zip(paths, row_groups_per_file, strict=True) for rg in range(n)}
    assert len(assigned) == len(expected), "row groups must not be duplicated across shards"
    assert set(assigned) == expected

    # Planning is deterministic.
    assert _build_stage2_shards(paths, shard_count) == (plans, total)


def test_build_stage2_shards_rejects_bad_counts(tmp_path: Path) -> None:
    path = tmp_path / "in.parquet"
    _write_parquet_with_row_groups(path, 2)
    with pytest.raises(ValueError):
        _build_stage2_shards([path], 0)
    with pytest.raises(ValueError):
        _build_stage2_shards([path], 3)


# --------------------------------------------------------------------------------------
# Stage 2: per-row FHR and toco derivation
# --------------------------------------------------------------------------------------


def test_compute_fhr_means_positive_hr1_values() -> None:
    batch = pa.RecordBatch.from_pydict(
        {
            "Hr1_0": pa.array([120.0, 0.0, None, 140.0, -5.0], type=pa.float64()),
            "Hr1_1": pa.array([0.0, 0.0, None, 141.0, 100.0], type=pa.float64()),
            "Hr1_2": pa.array([None, 0.0, None, 142.0, 0.0], type=pa.float64()),
            "Hr1_3": pa.array([130.0, 0.0, None, 143.0, None], type=pa.float64()),
        }
    )
    fhr = _compute_fhr(batch)
    assert fhr.type == pa.float32()
    assert fhr.to_pylist() == pytest.approx([125.0, 0.0, 0.0, 141.5, 100.0])


def test_compute_fhr_does_not_truncate_int16_inputs() -> None:
    # The converter writes Hr1_* as int16; the mean must not use integer division.
    batch = pa.RecordBatch.from_pydict(
        {
            "Hr1_0": pa.array([140, 0, None], type=pa.int16()),
            "Hr1_1": pa.array([141, 0, None], type=pa.int16()),
            "Hr1_2": pa.array([142, 0, None], type=pa.int16()),
            "Hr1_3": pa.array([143, 0, None], type=pa.int16()),
        }
    )
    fhr = _compute_fhr(batch)
    assert fhr.type == pa.float32()
    assert fhr.to_pylist() == pytest.approx([141.5, 0.0, 0.0])


def test_compute_toco_decodes_base64_and_averages_valid_bytes() -> None:
    def enc(raw: bytes) -> str:
        return base64.b64encode(raw).decode("ascii")

    batch = pa.RecordBatch.from_pydict(
        {
            "Toco_Values": pa.array(
                [
                    enc(bytes([10, 20, 30])),  # all in 1..99 -> 20
                    enc(bytes([0, 255, 50])),  # only 50 valid -> 50
                    enc(bytes([0, 255])),  # nothing valid -> plain mean 127.5
                    enc(b""),  # empty payload -> 0
                    None,  # null -> 0
                    "not base64!!",  # undecodable -> 0
                ],
                type=pa.string(),
            )
        }
    )
    toco = _compute_toco(batch)
    assert toco.type == pa.float32()
    assert toco.to_pylist() == pytest.approx([20.0, 50.0, 127.5, 0.0, 0.0, 0.0])


def test_compute_toco_without_column_is_all_zero() -> None:
    batch = pa.RecordBatch.from_pydict({"other": pa.array([1, 2, 3])})
    toco = _compute_toco(batch)
    assert toco.type == pa.float32()
    assert toco.to_pylist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------------------
# Stage 3: BabyID hashing and PatientID bucketing SQL expressions
# --------------------------------------------------------------------------------------

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture
def sessions_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE sessions (PatientID VARCHAR, session_end TIMESTAMP)")
    con.executemany(
        "INSERT INTO sessions VALUES (?, ?)",
        [
            ("19800101-1234", datetime(2020, 5, 1, 10, 0, 0)),
            ("19800101-1234", datetime(2020, 5, 1, 12, 0, 0)),
            ("19900202-5678", datetime(2020, 5, 1, 10, 0, 0)),
            ("ABC-XYZ", datetime(2020, 5, 1, 10, 0, 0)),
        ],
    )
    return con


def _baby_ids(con: duckdb.DuckDBPyConnection, salt: str) -> list[str]:
    expr = _stage3_babyid_expr("sha256", salt)
    rows = con.execute(f"SELECT {expr} FROM sessions ORDER BY PatientID, session_end").fetchall()
    return [row[0] for row in rows]


def test_stage3_babyid_expr_is_sha256_deterministic_and_salted(
    sessions_con: duckdb.DuckDBPyConnection,
) -> None:
    ids = _baby_ids(sessions_con, "salt-one")
    assert all(_HEX64.match(value) for value in ids)
    # Distinct (PatientID, session_end) pairs give distinct BabyIDs.
    assert len(set(ids)) == len(ids) == 4
    # Same salt -> same ids; different salt -> different ids.
    assert _baby_ids(sessions_con, "salt-one") == ids
    assert set(_baby_ids(sessions_con, "salt-two")).isdisjoint(ids)
    # Single quotes in the salt are escaped rather than breaking the SQL.
    assert all(_HEX64.match(value) for value in _baby_ids(sessions_con, "it's"))


def test_stage3_bucket_expr_uses_id_suffix_and_stays_in_range(
    sessions_con: duckdb.DuckDBPyConnection,
) -> None:
    bucket_count = 256
    expr = _stage3_bucket_expr(bucket_count)
    rows = sessions_con.execute(
        f"SELECT DISTINCT PatientID, {expr} AS b FROM sessions ORDER BY PatientID"
    ).fetchall()
    buckets = dict(rows)
    assert buckets["19800101-1234"] == 1234 % bucket_count
    assert buckets["19900202-5678"] == 5678 % bucket_count
    # Non-numeric suffix falls back to hash(), but must still land in 0..bucket_count-1.
    assert 0 <= buckets["ABC-XYZ"] < bucket_count
    assert all(isinstance(value, int) for value in buckets.values())
