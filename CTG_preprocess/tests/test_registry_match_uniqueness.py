"""Stage 7 match uniqueness: one registry row per BabyID and one BabyID per registry row."""

from __future__ import annotations

import duckdb
from registry_matching import (
    MULTI_BABY_REGISTRY_ROWS_SQL,
    MULTI_REGISTRY_BABIES_SQL,
    UNIQUE_MATCHES_SQL,
)


def _matches_table(rows: list[tuple[int, str, int]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE matches (reg_row INTEGER, BabyID VARCHAR, apgar5 INTEGER)")
    con.executemany("INSERT INTO matches VALUES (?, ?, ?)", rows)
    return con


def test_twins_sharing_one_babyid_are_dropped_not_duplicated() -> None:
    con = _matches_table(
        [
            (1, "baby_single", 9),
            (2, "baby_twins", 9),  # twin A
            (3, "baby_twins", 4),  # twin B, different Apgar, same mother/CTG
        ]
    )
    kept = con.execute(f"SELECT reg_row, BabyID FROM ({UNIQUE_MATCHES_SQL}) ORDER BY 1").fetchall()
    assert kept == [(1, "baby_single")]
    assert con.execute(MULTI_REGISTRY_BABIES_SQL).fetchone() == (1,)
    assert con.execute(MULTI_BABY_REGISTRY_ROWS_SQL).fetchone() == (0,)


def test_registry_row_matching_two_babyids_is_dropped() -> None:
    con = _matches_table(
        [
            (1, "baby_a", 9),
            (1, "baby_b", 9),  # same registry row matched two CTG episodes
            (2, "baby_c", 7),
        ]
    )
    kept = con.execute(f"SELECT reg_row, BabyID FROM ({UNIQUE_MATCHES_SQL})").fetchall()
    assert kept == [(2, "baby_c")]
    assert con.execute(MULTI_BABY_REGISTRY_ROWS_SQL).fetchone() == (1,)
    assert con.execute(MULTI_REGISTRY_BABIES_SQL).fetchone() == (0,)


def test_unique_matches_keeps_one_row_per_babyid() -> None:
    con = _matches_table([(i, f"baby_{i}", 9) for i in range(5)])
    rows = con.execute(f"SELECT BabyID FROM ({UNIQUE_MATCHES_SQL})").fetchall()
    assert len(rows) == 5
    assert len({r[0] for r in rows}) == 5
