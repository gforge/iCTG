#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX
from .json_decoder import dataframe_from_glob
from .write_parquet import write_parquet_per_input


def convert_raw_ctg() -> None:
    ap = argparse.ArgumentParser(
        description=(
            f"Stream-parse concatenated JSON from {JSON_SUFFIX}/{ZIP_SUFFIX}, validate with "
            "Pydantic, return DataFrame or write Parquet."
        )
    )
    ap.add_argument(
        "inputs", nargs="+", help="Input files or globs (e.g. 'Export_*.json')"
    )
    ap.add_argument(
        "--zip-member",
        default=None,
        help=f"JSON member name inside zip (optional; defaults to largest {JSON_SUFFIX})",
    )
    ap.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Preview first N rows as a DataFrame (prints .head())",
    )
    ap.add_argument(
        "--parquet-out",
        type=Path,
        default=None,
        help="Write one Parquet per input into this directory (constant memory)",
    )
    args = ap.parse_args()

    if args.preview:
        df = dataframe_from_glob(args.inputs, args.zip_member, limit=args.preview)
        print(df.head(args.preview).to_string(index=False))

    if args.parquet_out:
        if not args.parquet_out.exists():
            args.parquet_out.mkdir(parents=True, exist_ok=True)
        elif not args.parquet_out.is_dir():
            raise ValueError(f"Not a directory: {args.parquet_out}")

        write_parquet_per_input(args.inputs, args.parquet_out, args.zip_member)


if __name__ == "__main__":
    convert_raw_ctg()
