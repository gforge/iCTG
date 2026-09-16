from __future__ import annotations

import glob
import os

# Default paths and settings. The defaults point at the shared server layout under
# /srv/data/input/iCTG; override any of them with the environment variables named below
# (e.g. on a workstation with a local copy) instead of editing this file.

# Registry export root: the Swedish Pregnancy Register ("SPR data ...") directory and the
# SNQ xlsx, as delivered. Export names are long and carry an ID, so they are resolved by glob.
DEFAULT_REGISTRY_ROOT = os.environ.get(
    "CTG_REGISTRY_ROOT", "/srv/data/input/iCTG/CTG_registry_data"
)


def _find_one(pattern: str, fallback: str) -> str:
    hits = sorted(glob.glob(os.path.join(DEFAULT_REGISTRY_ROOT, pattern)))
    return hits[0] if hits else fallback


# SPR export directory: gravniva.csv, pop.csv and the dated diagnosis/procedure tables.
DEFAULT_SPR_DIR = os.environ.get(
    "CTG_SPR_DIR", _find_one("SPR data*", f"{DEFAULT_REGISTRY_ROOT}/SPR")
)
# CSV with patient metadata (gravniva.csv), one row per live-born singleton.
DEFAULT_PATIENT_CSV = os.environ.get("CTG_PATIENT_CSV", f"{DEFAULT_SPR_DIR}/gravniva.csv")
# Dated long tables (one row per diagnosis/procedure code) from the same export.
DEFAULT_MOTHER_DIAG_CSV = f"{DEFAULT_SPR_DIR}/fv1_moderns_diagnoser.csv"
DEFAULT_CHILD_DIAG_CSV = f"{DEFAULT_SPR_DIR}/barn_barnets_diagnoser_forsta_28_dagarna.csv"
DEFAULT_CHILD_PROC_CSV = f"{DEFAULT_SPR_DIR}/barn_barnets_atgarder_forsta_28_dagarna.csv"
# SNQ registry data (Excel or CSV).
DEFAULT_SNQ_FILE = os.environ.get(
    "CTG_SNQ_FILE", _find_one("SNQ data*.xlsx", f"{DEFAULT_REGISTRY_ROOT}/SNQ data.xlsx")
)
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
# Minimum seconds of non-zero FHR inside the final window for a pregnancy to be kept.
# 1200 s (20 min) until 2026-09-05; lowered to 600 s on the clinical lead's decision, the
# models see the padding mask so they know how much signal is missing.
DEFAULT_STAGE5_MIN_FHR_SECONDS = 600
DEFAULT_STAGE5_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_5_qualityfilter"
DEFAULT_STAGE5_OUTPUT_FILE = f"{DEFAULT_STAGE5_DIR}/stage5_quality.parquet"
DEFAULT_STAGE5_5_OUTPUT_FILE = f"{DEFAULT_STAGE5_DIR}/stage5_5_sorted.parquet"
DEFAULT_STAGE6_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_6_partitioned"
DEFAULT_STAGE7_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_7_registrymatching"
DEFAULT_STAGE7_REGISTRY_CSV = f"{DEFAULT_STAGE7_DIR}/registry.csv"
DEFAULT_STAGE7_CTG_PARQUET = f"{DEFAULT_STAGE7_DIR}/ctg_final.parquet"
# Stage 9: clinician events from the ExportSignatures files (converted with `ictg-signatures`).
DEFAULT_EVENTS_DIR = os.environ.get("CTG_EVENTS_DIR", "/srv/data/input/iCTG/parquet_events")
DEFAULT_STAGE9_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_9_events"
# RegistrationID -> PatientID and time span, derived once from stage 0 (slow scan, cached).
DEFAULT_STAGE9_REGISTRATION_MAP = f"{DEFAULT_STAGE9_DIR}/registration_map.parquet"
# Events linked to BabyID, still with the free text (intermediate, restricted).
DEFAULT_STAGE9_EVENTS = f"{DEFAULT_STAGE9_DIR}/events_linked.parquet"
# Time-shifted deliverable without free text or staff names.
DEFAULT_STAGE8_EVENTS_PARQUET = f"{DEFAULT_REDUCTION_ROOT}/stage_8_timeshift/events.parquet"
# Per-pregnancy features aggregated from the events up to the end of the final CTG window
# (no timestamps, so nothing to shift). Joined to registry.csv by BabyID in CTG_ML.
DEFAULT_STAGE8_EVENT_FEATURES_CSV = (
    f"{DEFAULT_REDUCTION_ROOT}/stage_8_timeshift/events_features.csv"
)
# A registration is assigned to a pregnancy when it starts within this margin of the
# pregnancy's session span (all sessions of the pregnancy).
DEFAULT_EVENT_LINK_MARGIN_HOURS = 24
# Keyword flags derived from the free-text notes (Swedish clinical shorthand); the raw
# text never leaves stage 9. Names become boolean columns `note_<name>`.
DEFAULT_EVENT_NOTE_FLAGS = {
    "bricanyl": r"bricanyl|terbutalin",
    "oxytocin": r"oxytocin|syntocinon|synto\b",
    "amniotomy": r"amniotomi|hinnspr",
    "scalp_sample": r"skalp|laktat|lactat",
    "epidural": r"epidural|\beda\b",
    "c_section": r"sectio|kejsarsnitt|\bsnitt",
    "vacuum_or_forceps": r"sugklocka|\bve\b|vakuum|tång",
    "induction": r"misoprostol|cytotec|induktion|induk",
    "fever_or_infection": r"feber|antibiotika|infektion",
    "meconium": r"mekonium|meconium",
    "pushing": r"kryst",
    "stimulation": r"stimul",
}
# Anonymized long tables (BabyID, day_offset relative to birth, code), one row per code.
DEFAULT_STAGE7_MOTHER_DIAG_CSV = f"{DEFAULT_STAGE7_DIR}/mother_diagnoses.csv"
DEFAULT_STAGE7_CHILD_DIAG_CSV = f"{DEFAULT_STAGE7_DIR}/child_diagnoses.csv"
DEFAULT_STAGE7_CHILD_PROC_CSV = f"{DEFAULT_STAGE7_DIR}/child_procedures.csv"

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

# Secrets (BabyID salt, time-shift secret) live outside git: environment variables
# CTG_BABYID_SALT / CTG_TIMESHIFT_SECRET, or files in this directory (see secrets_store.py).
DEFAULT_SECRETS_DIR = os.environ.get("CTG_SECRETS_DIR", f"{DEFAULT_REDUCTION_ROOT}/secrets")

# Stage 8: time shifting of the final outputs (see time_shift.py).
DEFAULT_STAGE8_DIR = f"{DEFAULT_REDUCTION_ROOT}/stage_8_timeshift"
DEFAULT_STAGE8_REGISTRY_CSV = f"{DEFAULT_STAGE8_DIR}/registry.csv"
DEFAULT_STAGE8_CTG_PARQUET = f"{DEFAULT_STAGE8_DIR}/ctg_final.parquet"
DEFAULT_STAGE8_ALL_SESSIONS_DIR = f"{DEFAULT_STAGE8_DIR}/all_sessions"
# Per-BabyID shift table (BabyID, shift_days). Re-identification aid: stays with the
# intermediate data, never with the deliverable.
DEFAULT_STAGE8_KEY_FILE = f"{DEFAULT_STAGE8_DIR}/timeshift_key.parquet"
# BabyID -> MotherID for every pregnancy in the CTG data (also the pretraining-only ones),
# so mother-level splits and leakage exclusions can be built downstream.
DEFAULT_STAGE8_MOTHERS_CSV = f"{DEFAULT_STAGE8_DIR}/mothers.csv"
# Every mother gets a base shift of whole days drawn uniformly from [-MAX, +MAX] ...
DEFAULT_TIMESHIFT_MAX_DAYS = 365
# ... and the interval between her consecutive pregnancies is stretched or shrunk by a
# factor drawn uniformly from [JITTER_MIN, JITTER_MAX], with random sign.
DEFAULT_TIMESHIFT_JITTER_MIN = 0.10
DEFAULT_TIMESHIFT_JITTER_MAX = 0.20
# registry.csv columns that carry absolute dates/timestamps and must be shifted.
DEFAULT_TIMESHIFT_REGISTRY_COLUMNS = [
    "birth_day",
    "birth_timestamp",
    "etablerade_varkar_datum",
    "etablerade_varkar_timestamp",
    "avled_datum",
]

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
    "fhr_stv",
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
