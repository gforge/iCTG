from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def verify_parquet(path: Path, sample_rate_hz: int) -> None:
    table = pq.read_table(path)
    df = table.to_pandas()

    expected_rows = 3600 if sample_rate_hz == 1 else 14400
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns: {list(df.columns)}")

    if "BabyID" not in df.columns:
        print("Missing BabyID column.")
        return

    counts = df.groupby("BabyID")["Timestamp"].count()
    bad = counts[counts != expected_rows]
    print(f"Unique BabyIDs: {counts.size}")
    print(f"Expected rows per BabyID: {expected_rows}")
    print(f"BabyIDs with wrong row count: {len(bad)}")
    if not bad.empty:
        print(bad.head(10).to_string())

    ts_min = df["Timestamp"].min()
    ts_max = df["Timestamp"].max()
    print(f"Timestamp range: {ts_min} to {ts_max}")

    fhr_nan = df["FHR"].isna().mean()
    print(f"FHR NaN ratio: {fhr_nan:.2%}")

    if "toco" in df.columns:
        toco_nan = df["toco"].isna().mean()
        print(f"Toco NaN ratio: {toco_nan:.2%}")

    sample = df[df["BabyID"] == counts.index[0]].sort_values("Timestamp")
    if not sample.empty:
        diffs = pd.Series(sample["Timestamp"]).diff().dropna()
        print(f"Sample BabyID spacing: {diffs.mode().iloc[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CTG output parquet.")
    parser.add_argument("parquet_path", type=str, help="Path to parquet file.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[1, 4],
        default=1,
        help="Sample rate used when generating the file.",
    )
    args = parser.parse_args()
    verify_parquet(Path(args.parquet_path), args.sample_rate)


if __name__ == "__main__":
    main()
