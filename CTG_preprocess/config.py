from __future__ import annotations

import os

# Default paths and settings. The defaults point at the shared server layout under
# /srv/data/input/iCTG; override any of them with the environment variables named below
# (e.g. on a workstation with a local copy) instead of editing this file.

# CSV with patient metadata (gravniva.csv).
DEFAULT_PATIENT_CSV = os.environ.get(
    "CTG_PATIENT_CSV", "/srv/data/input/iCTG/registry/gravniva.csv"
)
# SNQ registry data (Excel or CSV).
DEFAULT_SNQ_FILE = os.environ.get("CTG_SNQ_FILE", "/srv/data/input/iCTG/registry/SNQ data.xlsx")
# Root directory for staged data reduction outputs.
DEFAULT_REDUCTION_ROOT = os.environ.get(
    "CTG_REDUCTION_ROOT", "/srv/data/input/iCTG/processed/reduction"
)
# Raw CTG parquet input directory (converter output). Stage 1 reads every parquet file here.
DEFAULT_STAGE0_DIR = os.environ.get("CTG_STAGE0_DIR", "/srv/data/input/iCTG/parquet")
# Stage directories (derived from DEFAULT_REDUCTION_ROOT).
DEFAULT_STAGE1_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_1_timefilter"
DEFAULT_STAGE2_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_2_columnfilter"
DEFAULT_STAGE2_EXTRA_COLUMNS = [
    "Hr1_SignalQuality",
    "Hr1Mode",
    "TocoMode",
]
DEFAULT_STAGE3_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_3_sessionfilter"
DEFAULT_STAGE4_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_4_duplicatefilter"
DEFAULT_STAGE4_OUTPUT_FILE = f"{DEFAULT_STAGE4_DIR}/stage4_dedup.parquet"
DEFAULT_STAGE4_DUP_THRESHOLD = 0.30
DEFAULT_STAGE5_MIN_FHR_SECONDS = 1200
DEFAULT_STAGE5_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_5_qualityfilter"
DEFAULT_STAGE5_OUTPUT_FILE = f"{DEFAULT_STAGE5_DIR}/stage5_quality.parquet"
DEFAULT_STAGE5_5_OUTPUT_FILE = f"{DEFAULT_STAGE5_DIR}/stage5_5_sorted.parquet"
DEFAULT_STAGE6_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_6_partitioned"
DEFAULT_STAGE7_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_7_registrymatching"
DEFAULT_STAGE7_REGISTRY_CSV = f"{DEFAULT_STAGE7_DIR}/registry.csv"
DEFAULT_STAGE7_CTG_PARQUET = f"{DEFAULT_STAGE7_DIR}/ctg_final.parquet"

DEFAULT_STAGE3_OUTPUT_FILE = f"{DEFAULT_STAGE3_DIR}/stage3_sessions.parquet"

# Stage 3 session filter settings.
DEFAULT_STAGE3_GAP_MINUTES = 5
DEFAULT_STAGE3_PREG_GAP_DAYS = 200
DEFAULT_STAGE3_LAST_HOUR_MINUTES = 60
# How the final window is anchored:
#   "pregnancy"     - last non-zero FHR across ALL sessions of the pregnancy; the window may
#                     span several sessions (transfers to theatre etc. no longer lose the labour).
#   "final_session" - legacy behaviour: only rows from the last session are considered.
DEFAULT_STAGE3_WINDOW_SCOPE = "pregnancy"
# Optional Stage 3 side output with ALL sessions of every pregnancy (no PatientID), used
# for self-supervised pretraining. Written only when --stage3-all-sessions-out is given.
DEFAULT_STAGE3_ALL_SESSIONS_DIR = f"{DEFAULT_STAGE3_DIR}/all_sessions"
DEFAULT_BABYID_SALT = "VibeSaltTemp123"

# Progress report frequency (patients). Set to 0 to disable.
DEFAULT_REPORT_EVERY = 1000

# Stage 1 time cutoff (YYYY-MM-DD). Rows before this are dropped.
DEFAULT_STAGE1_CUTOFF_DATE = "2014-12-31"

# Partitioning defaults (final stage).
# Where to write the partitioned dataset (can be absolute).
# DEFAULT_PARTITION_OUTPUT_DIR = DEFAULT_STAGE6_DIR
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
    *DEFAULT_STAGE2_EXTRA_COLUMNS,
]

# Use the partitioned dataset for main processing by default.
DEFAULT_USE_PARTITIONED_DATASET = True
# Partitioning progress reporting (every N batches). Set to 0 to disable.
DEFAULT_PARTITION_REPORT_EVERY = 50
# Number of patient buckets to partition by (power of 2 recommended, e.g. 256).
DEFAULT_PARTITION_BUCKETS = 256
# Stage 3 bucketing (set >1 to process in smaller chunks and avoid OOM).
DEFAULT_STAGE3_BUCKETS = 256
