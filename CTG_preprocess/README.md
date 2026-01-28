CTG Preprocess

Quick Start
- Test mode (find first valid patient in a range):
  `uv run python main.py --test-first-valid`
- Full processing (writes metadata + parquet output):
  `uv run python main.py --process --start 1 --limit 1000 --batch-id 1 --output-dir output`

Inputs (edit defaults in config.py)
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
