"""Execute the registry-cleaning SQL expression builders against tiny in-memory DuckDB tables."""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import pytest
from registry_matching import (
    _bool_ja_nej_expr,
    _code_exact_expr,
    _code_prefix_expr,
    _int_expr,
    _normalized_glopnr_expr,
)


def _eval(expr: str, column: str, values: Sequence[object]) -> list[object]:
    """Evaluate ``expr`` (which references ``column``) once per input value, preserving order."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t (i INTEGER, {column} VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
    return [row[0] for row in con.execute(f"SELECT {expr} FROM t ORDER BY i").fetchall()]


def test_normalized_glopnr_strips_float_suffix_and_whitespace() -> None:
    values = ["12.0", "12.000", " 1 2 ", "3.50", "", "   ", None]
    assert _eval(_normalized_glopnr_expr("glopnr"), "glopnr", values) == [
        "12",
        "12",
        "12",
        "3.50",
        None,
        None,
        None,
    ]


def test_bool_ja_nej_expr_maps_swedish_yes_no() -> None:
    values = ["Ja", "ja ", " NEJ", "nej", "", "kanske", None]
    assert _eval(_bool_ja_nej_expr("flag"), "flag", values) == [
        True,
        True,
        False,
        False,
        None,
        None,
        None,
    ]


def test_int_expr_accepts_decimal_comma_and_rejects_garbage() -> None:
    values = ["7", " 8 ", "9,0", "abc", "", None]
    assert _eval(_int_expr("apgar"), "apgar", values) == [7, 8, 9, None, None, None]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (_code_prefix_expr("diag", ["O13", "O16"]), [True, True, False, True, False, False]),
        (_code_exact_expr("diag", ["O710", "O711"]), [False, True, True, False, False, False]),
    ],
)
def test_code_matching_over_comma_separated_lists(expr: str, expected: list[bool]) -> None:
    values = [
        "O139, Z000",  # prefix hit (space after comma is stripped)
        "o710,O160",  # lower-case is upper-cased; exact hit + prefix hit
        "Z00,O711",  # exact hit only
        "O13",  # bare prefix code
        "O7100",  # longer code must not count as exact O710
        "",  # nothing
    ]
    assert _eval(expr, "diag", values) == expected
