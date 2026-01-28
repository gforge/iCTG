from __future__ import annotations

# Default paths and settings. Edit these on the server instead of main.py.

# CSV with patient metadata.
DEFAULT_PATIENT_CSV = "/home/lukas-uggla/Documents/Data/gravniva.csv"
# Parquet files with CTG data (can be absolute paths).
DEFAULT_PARQUET_PATHS = [
    "/home/lukas-uggla/Documents/Data/Export_2025-05-22 10_14_42.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-05-22 10_27_33.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-05-24 14_11_28.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-05-26 07_40_48.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-05-27 06_09_16.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-05-29 15_41_03.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-05-31 16_07_42.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-06-02 13_00_16.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-06-03 23_02_53.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-06-08 11_07_58.parquet",
    "/home/lukas-uggla/Documents/Data/Export_2025-06-10 09_23_30.parquet"
]
'''
[
    "../../../Data/Export_2025-05-22 10_27_33.parquet",
    "../../../Data/Export_2025-06-08 11_07_58.parquet",
]
'''

# Processing defaults
# 1 Hz or 4 Hz sampling.
DEFAULT_SAMPLE_RATE_HZ = 1
# How to reduce 4 samples/sec to 1 Hz when sample rate is 1. (mean or first)
DEFAULT_DOWNSAMPLE_MODE = "mean"
# Output mode: "dataset" writes a new parquet file per run, "append" grows one file.
DEFAULT_OUTPUT_MODE = "dataset"
# Max size for append mode before auto-switching to dataset mode. (Probably not needed)
DEFAULT_MAX_BATCH_SIZE_GB = 5.0
# Output directory path.
DEFAULT_OUTPUT_DIR = "output"
# Progress report frequency (patients). Set to 0 to disable.
DEFAULT_REPORT_EVERY = 1000

# Partitioning defaults (for the one-time preprocessing step).
# Where to write the partitioned dataset (can be absolute).
DEFAULT_PARTITION_OUTPUT_DIR = "/home/lukas-uggla/Documents/Data/ctg_partitioned"
# Drop any CTG rows before this date (YYYY-MM-DD).
DEFAULT_PARTITION_CUTOFF_DATE = "2014-12-24"
# Columns to keep in the partitioned dataset.
DEFAULT_PARTITION_COLUMNS = [
    "PatientID",
    "Timestamp",
    "Hr1_0",
    "Hr1_1",
    "Hr1_2",
    "Hr1_3",
    "Toco_Values",
]

# Use the partitioned dataset for main processing by default.
DEFAULT_USE_PARTITIONED_DATASET = True
# Partitioning progress reporting (every N batches). Set to 0 to disable.
DEFAULT_PARTITION_REPORT_EVERY = 50
# Number of patient buckets to partition by (power of 2 recommended, e.g. 256).
DEFAULT_PARTITION_BUCKETS = 256
