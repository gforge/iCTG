from __future__ import annotations

# Default paths and settings. Edit these on the server instead of main.py.

# CSV with patient metadata.
DEFAULT_PATIENT_CSV = "/home/lukas-uggla/Documents/Data/gravniva.csv"
# Raw parquet files with CTG data (can be absolute paths).
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
# Root directory for staged data reduction outputs.
DEFAULT_REDUCTION_ROOT = "/home/lukas-uggla/Documents/Data/ctg-data-reduction"
# Stage directories (derived from DEFAULT_REDUCTION_ROOT).
DEFAULT_STAGE1_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_1_timefilter"
DEFAULT_STAGE2_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_2_columnfilter"
DEFAULT_STAGE3_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_3_sessionfilter"
DEFAULT_STAGE4_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_4_duplicatefilter"
DEFAULT_STAGE4_OUTPUT_FILE = f"{DEFAULT_STAGE4_DIR}/stage4_dedup.parquet"
DEFAULT_STAGE4_DUP_THRESHOLD = 0.30
DEFAULT_STAGE5_MIN_FHR_SECONDS = 1200
DEFAULT_STAGE5_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_5_qualityfilter"
DEFAULT_STAGE5_OUTPUT_FILE = f"{DEFAULT_STAGE5_DIR}/stage5_quality.parquet"
DEFAULT_STAGE5_5_OUTPUT_FILE = f"{DEFAULT_STAGE5_DIR}/stage5_5_sorted.parquet"
DEFAULT_STAGE6_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_6_partitioned"

DEFAULT_STAGE3_OUTPUT_FILE = f"{DEFAULT_STAGE3_DIR}/stage3_sessions.parquet"

# Stage 3 session filter settings.
DEFAULT_STAGE3_GAP_MINUTES = 5
DEFAULT_STAGE3_PREG_GAP_DAYS = 200
DEFAULT_STAGE3_LAST_HOUR_MINUTES = 60
DEFAULT_BABYID_SALT = "VibeSaltTemp123"

# Progress report frequency (patients). Set to 0 to disable.
DEFAULT_REPORT_EVERY = 1000

# Stage 1 time cutoff (YYYY-MM-DD). Rows before this are dropped.
DEFAULT_STAGE1_CUTOFF_DATE = "2014-12-31"

# Partitioning defaults (final stage).
# Where to write the partitioned dataset (can be absolute).
#DEFAULT_PARTITION_OUTPUT_DIR = DEFAULT_STAGE6_DIR
DEFAULT_PARTITION_OUTPUT_DIR = DEFAULT_STAGE6_DIR
# Drop any CTG rows before this date (YYYY-MM-DD).
DEFAULT_PARTITION_CUTOFF_DATE = DEFAULT_STAGE1_CUTOFF_DATE
# Columns to keep in the partitioned dataset.
DEFAULT_PARTITION_COLUMNS = [
    "BabyID",
    "PatientID",
    "Timestamp",
    "FHR",
    "toco",
]

# Use the partitioned dataset for main processing by default.
DEFAULT_USE_PARTITIONED_DATASET = True
# Partitioning progress reporting (every N batches). Set to 0 to disable.
DEFAULT_PARTITION_REPORT_EVERY = 50
# Number of patient buckets to partition by (power of 2 recommended, e.g. 256).
DEFAULT_PARTITION_BUCKETS = 256
# Stage 3 bucketing (set >1 to process in smaller chunks and avoid OOM).
DEFAULT_STAGE3_BUCKETS = 256
