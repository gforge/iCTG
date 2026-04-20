#!/usr/bin/env python3
"""CLI entrypoint for streaming raw CTG JSON exports into parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX
from .json_decoder import (
    FailureTracker,
    dataframe_from_glob,
    format_failure_detail,
    format_failure_summary,
)
from .logging_config import setup_logging
from .write_parquet import write_parquet_per_input


DETAIL_LOG_THRESHOLD = 20
TOP_PROBLEM_FILES = 5


def _get_log_file_path(logger: logging.Logger) -> Path | None:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def _log_to_file_handlers(
    logger: logging.Logger, level: int, message: str, *, pathname: str = __file__
) -> None:
    record = logger.makeRecord(
        logger.name,
        level,
        pathname,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def _log_failures(logger: logging.Logger, failure_tracker: FailureTracker) -> None:
    if failure_tracker.count == 0:
        return

    details = [format_failure_detail(failure) for failure in failure_tracker.failures()]
    log_path = _get_log_file_path(logger)

    if failure_tracker.count < DETAIL_LOG_THRESHOLD:
        logger.warning(
            "Encountered %s conversion failures", f"{failure_tracker.count:,}"
        )
        for detail in details:
            logger.warning(detail)
        return

    if log_path is not None:
        for detail in details:
            _log_to_file_handlers(logger, logging.WARNING, detail)
    else:
        for detail in details:
            logger.warning(detail)

    for line in format_failure_summary(
        failure_tracker.count,
        failure_tracker.grouped_counts(),
        top_n=TOP_PROBLEM_FILES,
    ):
        logger.warning(line)

    if log_path is not None:
        logger.warning("See full failure details in log file: %s", log_path)


def convert_raw_ctg() -> None:
    """Parse CLI arguments and run preview and/or parquet conversion."""
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
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inputs whose final parquet output already exists.",
    )
    ap.add_argument(
        "--parquet-batch-size",
        type=int,
        default=50_000,
        help="Maximum records kept in memory before flushing a parquet batch.",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to store log files (default: data/log/)",
    )
    args = ap.parse_args()

    # Set up logging
    log_dir = Path("data/log")
    if args.parquet_out is not None:
        log_dir = args.parquet_out
    if args.log_dir is not None:
        log_dir = args.log_dir
    logger = setup_logging(log_dir=log_dir)
    failure_tracker = FailureTracker()
    logger.info("Starting conversion with args: %s", args)
    log_path = _get_log_file_path(logger)
    if log_path is not None:
        logger.info("Active log file: %s", log_path)

    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet-batch-size must be greater than 0")

    if args.preview:
        df = dataframe_from_glob(
            args.inputs,
            args.zip_member,
            limit=args.preview,
            failure_tracker=failure_tracker,
        )
        print(df.head(args.preview).to_string(index=False))

    if args.parquet_out:
        if not args.parquet_out.exists():
            args.parquet_out.mkdir(parents=True, exist_ok=True)
            logger.info("Created output directory: %s", args.parquet_out)
        elif not args.parquet_out.is_dir():
            raise ValueError(f"Not a directory: {args.parquet_out}")

        write_parquet_per_input(
            args.inputs,
            args.parquet_out,
            args.zip_member,
            batch_size=args.parquet_batch_size,
            skip_existing=args.skip_existing,
            failure_tracker=failure_tracker,
        )
        logger.info("Conversion completed successfully")

    _log_failures(logger, failure_tracker)


if __name__ == "__main__":
    convert_raw_ctg()
