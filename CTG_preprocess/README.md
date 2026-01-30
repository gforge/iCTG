CTG Preprocess

Quick Start
- Test mode (find first valid patient in a range):
  `uv run python main.py --test-first-valid`
- Full processing (writes metadata + parquet output):
  `uv run python main.py --process --start 1 --limit 1000 --batch-id 1 --output-dir output`

Inputs (edit defaults in config.py)
- Note: if DEFAULT_USE_PARTITIONED_DATASET is true, --parquet defaults to the partitioned output dir.
- `--patient-csv PATH` Path to patient CSV (default from `DEFAULT_PATIENT_CSV` in code).
- `--parquet PATH [PATH ...]` One or more CTG parquet files (default from `DEFAULT_PARQUET_PATHS`).

Range / Chunking
- `--start N` 1-based patient index to start from.
- `--limit N` Number of patients to scan/process.

Modes
- `--test-first-valid` Scan a range and stop at the Nth valid patient.
- `--process` Run full processing, write metadata + parquet output.

Signal settings
- `--sample-rate {1,4}` Process at 1 Hz or 4 Hz (default 1 Hz).
- `--downsample-mode {mean,first}` How to reduce 4 samples/second to 1 Hz (default mean).
- `--min-data FLOAT` Minimum fraction of valid data required (default 0.5).

Output (edit defaults in config.py)
- `--output-dir PATH` Directory for `metadata.csv` and parquet output (default from `DEFAULT_OUTPUT_DIR`).
- `--batch-id N` Batch number for file naming.
- `--output-mode {append,dataset}` Append to a batch file or write a new dataset file (default dataset).
- `--max-batch-size-gb FLOAT` Auto-switch to dataset output when append file exceeds this size (default 5.0).
- `--report-every N` Print progress every N patients.

Test-only extras
- `--valid-index N` Choose the Nth valid patient within the scan range.
- `--plot` Save a plot of FHR + toco.
- `--plot-out PATH` Output path for the plot image.
- `--print-limit N` Number of rows to print from start/end of the filtered signal.

Partitioning Step (recommended)
- One-time preprocessing to speed up patient lookups by splitting CTG data by date and patient bucket.
- Edit defaults in `config.py` (output dir, cutoff date, columns, bucket count).
- Run:
  `python partition_ctg.py --output-dir /path/to/ctg_partitioned`
- Optional: `--report-every-batches 50` for progress updates
- Then keep `DEFAULT_USE_PARTITIONED_DATASET = True` and set `DEFAULT_PARTITION_OUTPUT_DIR` in `config.py` (or pass `--parquet` at runtime).


Reduction Stages
- Run stage 1 (time filter):
  `uv run python ctg_reduction.py --stage stage1`
- Run stage 2 (column filter):
  `uv run python ctg_reduction.py --stage stage2`
- Stage 3 (session filter) is not implemented yet.
- Final partitioning (after stage 3):
  `uv run python ctg_reduction.py --stage partition`

Set paths in `config.py` (DEFAULT_REDUCTION_ROOT, stage dirs, cutoff date).
