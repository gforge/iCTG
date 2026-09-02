"""Stage 3 window selection and the all-sessions pretraining export on synthetic sessions.

Scenario: labour recordings are often fragmented (transfer to theatre, repositioning), so the
last *session* can be a short reconnection or a signal-less tail. The pregnancy-scoped window
must recover the preceding session; the legacy final-session scope must not (it documents the
behaviour that silently lost these pregnancies).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest
from ctg_reduction import (
    _stage3_all_sessions_query,
    _stage3_babyid_expr,
    _stage3_query,
    _stage4_query,
)

T0 = datetime(2021, 3, 1, 12, 0, 0)
SALT = "unit-test-salt"
MIN = timedelta(minutes=1)
Row = tuple[str, int, datetime, float, float, str, str, str]


def _session(
    patient: str, start: datetime, end: datetime, fhr: float, quality: str = "G"
) -> list[Row]:
    rows: list[Row] = []
    ts = start
    while ts <= end:
        rows.append((patient, 1, ts, fhr, 20.0, quality, "US", "TOCO"))
        ts += timedelta(seconds=1)
    return rows


def _connect(rows: list[Row]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE ctg (
            PatientID VARCHAR, RegistrationID INTEGER, Timestamp TIMESTAMP,
            FHR FLOAT, toco FLOAT, Hr1_SignalQuality VARCHAR, Hr1Mode VARCHAR, TocoMode VARCHAR
        )
        """
    )
    con.executemany("INSERT INTO ctg VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return con


def _query(scope: str, extra_where: str = "") -> str:
    return _stage3_query(
        _stage3_babyid_expr("sha256", SALT),
        gap_minutes=5,
        preg_gap_days=200,
        last_hour_minutes=60,
        extra_where=extra_where,
        window_scope=scope,
    )


def _all_sessions_query() -> str:
    return _stage3_all_sessions_query(
        _stage3_babyid_expr("sha256", SALT),
        gap_minutes=5,
        preg_gap_days=200,
        last_hour_minutes=60,
    )


@pytest.fixture
def fragmented_labour() -> list[Row]:
    """Antenatal visit 30 days earlier, a 90-min labour session, a 6-min gap, a 4-min tail."""
    rows = _session("P1", T0 - timedelta(days=30), T0 - timedelta(days=30) + 20 * MIN, 140.0)
    rows += _session("P1", T0 - 100 * MIN, T0 - 10 * MIN, 140.0)
    rows += _session("P1", T0 - 4 * MIN, T0, 138.0)
    return rows


def test_pregnancy_scope_spans_sessions_and_keeps_the_labour(
    fragmented_labour: list[Row],
) -> None:
    con = _connect(fragmented_labour)
    n_rows, first, last = con.execute(
        f"SELECT COUNT(*), MIN(Timestamp), MAX(Timestamp) FROM ({_query('pregnancy')})"
    ).fetchone()  # type: ignore[misc]
    # Window [T0-60min, T0]: 50 min + 1 s from the labour session, 4 min + 1 s from the tail.
    assert (first, last) == (T0 - 60 * MIN, T0)
    assert n_rows == (50 * 60 + 1) + (4 * 60 + 1)


def test_final_session_scope_only_keeps_the_short_tail(fragmented_labour: list[Row]) -> None:
    con = _connect(fragmented_labour)
    (n_rows,) = con.execute(f"SELECT COUNT(*) FROM ({_query('final_session')})").fetchone()  # type: ignore[misc]
    assert n_rows == 4 * 60 + 1  # would be dropped by Stage 5 (< 1200 s)


def test_both_scopes_assign_the_same_babyid(fragmented_labour: list[Row]) -> None:
    con = _connect(fragmented_labour)
    ids = {
        scope: con.execute(f"SELECT DISTINCT BabyID FROM ({_query(scope)})").fetchall()
        for scope in ("pregnancy", "final_session")
    }
    assert ids["pregnancy"] == ids["final_session"]
    assert len(ids["pregnancy"]) == 1


def test_pregnancy_scope_ignores_signal_less_final_session() -> None:
    rows = _session("P2", T0 - 120 * MIN, T0 - 15 * MIN, 135.0)
    rows += _session("P2", T0 - 5 * MIN, T0, 0.0)  # monitor left on, no signal
    con = _connect(rows)
    n_preg, last_preg = con.execute(
        f"SELECT COUNT(*), MAX(Timestamp) FROM ({_query('pregnancy')})"
    ).fetchone()  # type: ignore[misc]
    assert last_preg == T0 - 15 * MIN
    assert n_preg == 60 * 60 + 1
    (nz_final,) = con.execute(
        f"SELECT COUNT(*) FILTER (WHERE FHR > 0) FROM ({_query('final_session')})"
    ).fetchone()  # type: ignore[misc]
    assert nz_final == 0


def test_antenatal_visit_outside_pregnancy_gap_gets_its_own_babyid() -> None:
    rows = _session("P3", T0 - timedelta(days=300), T0 - timedelta(days=300) + 30 * MIN, 140.0)
    rows += _session("P3", T0 - 70 * MIN, T0, 140.0)
    con = _connect(rows)
    (n_babies,) = con.execute(
        f"SELECT COUNT(DISTINCT BabyID) FROM ({_query('pregnancy')})"
    ).fetchone()  # type: ignore[misc]
    assert n_babies == 2


def test_all_sessions_export_numbers_sessions_flags_window_and_hides_patientid(
    fragmented_labour: list[Row],
) -> None:
    duplicate = fragmented_labour[-1]
    con = _connect(fragmented_labour + [duplicate])
    con.execute(f"CREATE TABLE all_sessions AS {_all_sessions_query()}")
    con.execute(f"CREATE TABLE final_window AS {_query('pregnancy')}")

    columns = [r[0] for r in con.execute("DESCRIBE all_sessions").fetchall()]
    assert "PatientID" not in columns
    assert columns[:3] == ["BabyID", "session_id", "Timestamp"]

    sessions = con.execute(
        "SELECT session_id, COUNT(*), bool_or(in_final_window), bool_and(in_final_window) "
        "FROM all_sessions GROUP BY session_id ORDER BY session_id"
    ).fetchall()
    assert [s[0] for s in sessions] == [1, 2, 3]
    assert sessions[0][1] == 20 * 60 + 1 and sessions[0][2] is False  # antenatal
    assert sessions[1][2] is True and sessions[1][3] is False  # labour: partly in window
    assert sessions[2][1] == 4 * 60 + 1 and sessions[2][3] is True  # tail fully in window

    # Same BabyID as the supervised window; duplicate timestamps collapsed.
    assert con.execute(
        "SELECT COUNT(DISTINCT BabyID) FROM all_sessions "
        "WHERE BabyID IN (SELECT BabyID FROM final_window)"
    ).fetchone() == (1,)
    (in_window,) = con.execute("SELECT COUNT(*) FROM all_sessions WHERE in_final_window").fetchone()  # type: ignore[misc]
    # The re-exported (identical) row is collapsed in both outputs.
    assert in_window == con.execute("SELECT COUNT(*) FROM final_window").fetchone()[0]  # type: ignore[index]
    assert in_window == (50 * 60 + 1) + (4 * 60 + 1)


def test_stage4_ignores_exact_duplicates_but_counts_conflicts() -> None:
    def rows(baby: str, fhr_b: float | None) -> list[tuple[object, ...]]:
        out: list[tuple[object, ...]] = []
        for i in range(10):
            ts = T0 + timedelta(seconds=i)
            out.append((baby, "P", ts, 140.0, 20.0, "G", "US", "TOCO"))
            if fhr_b is not None:
                out.append((baby, "P", ts, fhr_b, 20.0, "G", "US", "TOCO"))
        return out

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE ctg (
            BabyID VARCHAR, PatientID VARCHAR, Timestamp TIMESTAMP, FHR FLOAT, toco FLOAT,
            Hr1_SignalQuality VARCHAR, Hr1Mode VARCHAR, TocoMode VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO ctg VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows("exact_dup", 140.0) + rows("conflict", 90.0) + rows("clean", None),
    )
    kept = con.execute(
        f"SELECT BabyID, COUNT(*), MIN(FHR) FROM ({_stage4_query(0.30)}) GROUP BY BabyID ORDER BY 1"
    ).fetchall()
    assert kept == [("clean", 10, 140.0), ("exact_dup", 10, 140.0)]
