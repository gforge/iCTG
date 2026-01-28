from __future__ import annotations

# Default paths and settings. Edit these on the server instead of main.py.

# CSV with patient metadata.
DEFAULT_PATIENT_CSV = "../../Data/gravniva.csv"
# Parquet files with CTG data (can be absolute paths).
DEFAULT_PARQUET_PATHS = [
    "../../Data/Export_2025-05-22 10_27_33.parquet",
    "../../Data/Export_2025-06-08 11_07_58.parquet",
]

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
