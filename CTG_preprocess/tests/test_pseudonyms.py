"""The DuckDB and Python MotherID derivations must agree."""

from __future__ import annotations

import duckdb

from pseudonyms import MOTHER_ID_LENGTH, mother_id, mother_id_sql


def test_sql_and_python_mother_id_agree() -> None:
    salt = "sa'lt"
    con = duckdb.connect()
    con.execute("CREATE TABLE t (PatientID VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?)", [("19800101-0001",), ("x'y",)])
    expr = mother_id_sql(salt, "PatientID")
    rows = con.execute(f"SELECT PatientID, {expr} FROM t").fetchall()
    assert len(rows) == 2
    for patient, sql_value in rows:
        assert sql_value == mother_id(salt, patient)
        assert len(sql_value) == MOTHER_ID_LENGTH
    assert mother_id("a", "p") != mother_id("b", "p")
