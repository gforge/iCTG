"""DuckDB connections for the preprocessing steps.

DuckDB spills to a temp directory when a query exceeds memory. The default (``.tmp`` under
the working directory) sits on the small system disk here, so ``CTG_DUCKDB_TEMP_DIR`` can
point it at the data disk. Set ``CTG_DUCKDB_MEMORY_LIMIT`` (e.g. ``"48GB"``) to cap memory.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    temp_dir = os.environ.get("CTG_DUCKDB_TEMP_DIR")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{temp_dir.replace(chr(39), chr(39) * 2)}'")
    memory_limit = os.environ.get("CTG_DUCKDB_MEMORY_LIMIT")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    return con
