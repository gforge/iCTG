# CTG Preprocess

Pipeline for reducing raw CTG parquet data, matching the reduced CTG cohort to registry data, and writing anonymized final outputs.

## Main scripts

- `ctg_reduction.py`: Stages 1-6 of CTG reduction.
- `registry_matching.py`: Stage 7 registry matching and anonymized export.
- `config.py`: all input paths, output paths, and stage settings.

## Environment

This project is run with `uv` and Python 3.12.

Install dependencies:

```bash
uv sync
```

If you prefer `pip`, install from `requirements.txt`.

## Configuration

`config.py` defaults to the shared-server layout under `/srv/data/input/iCTG`. Override the
paths with environment variables instead of editing the file:

| Setting | Environment variable | Default |
|---------|----------------------|---------|
| Raw CTG parquet from the converter (stage 0) | `CTG_STAGE0_DIR` | `/srv/data/input/iCTG/parquet` |
| Main registry file (`gravniva.csv`) | `CTG_PATIENT_CSV` | `/srv/data/input/iCTG/registry/gravniva.csv` |
| SNQ registry file | `CTG_SNQ_FILE` | `/srv/data/input/iCTG/registry/SNQ data.xlsx` |
| Root for stage 1-7 outputs | `CTG_REDUCTION_ROOT` | `/srv/data/input/iCTG/processed/reduction` |

Stage settings (session gap, window length, thresholds, BabyID salt) are also in `config.py`.

## Pipeline

Run everything (stages 1-7 plus the cohort and match-loss reports) inside tmux:

```bash
tmux new -s ctg-pipeline 'CTG_preprocess/run_pipeline.sh'
# resume from a stage: START_STAGE=stage3 CTG_preprocess/run_pipeline.sh
```

Or run the stages individually, in order:

```bash
uv run python ctg_reduction.py --stage stage1
uv run python ctg_reduction.py --stage stage2
uv run python ctg_reduction.py --stage stage3 --stage3-all-sessions-out
uv run python ctg_reduction.py --stage stage4
uv run python ctg_reduction.py --stage stage5
uv run python ctg_reduction.py --stage stage5_5
uv run python ctg_reduction.py --stage stage6
uv run python registry_matching.py
```

Stage semantics that matter for cohort size (see `docs/cohort_loss_audit.md` for the numbers
behind them):

- Stage 3 collapses identical rows first: the raw exports overlap in time, so most recordings
  occur in several export files.
- Stage 3 anchors the 60-minute window on the last non-zero FHR of the *whole pregnancy*
  (`--stage3-window-scope pregnancy`, default), so a fragmented labour (transfer to theatre,
  brief reconnection, signal-less tail) keeps its recording. `--stage3-window-scope
  final_session` restores the old behaviour of only looking at the last session.
- `--stage3-all-sessions-out [DIR]` additionally writes every session of every pregnancy
  (BabyID, `session_id`, `in_final_window`, no PatientID) under
  `DEFAULT_STAGE3_ALL_SESSIONS_DIR` for self-supervised pretraining in `CTG_ML`.
- Stage 4 drops a BabyID only when more than 30 % of its timestamps have *conflicting*
  values; exact duplicates are merged.
- Stage 7 keeps one registry row per BabyID and one BabyID per registry row; twins and
  multiples are excluded rather than duplicated.

Stage 3 defaults to a one-pass temporary PatientID pre-bucketing step before it runs the session logic. This avoids rescanning the full Stage 2 output once per bucket, but it temporarily needs disk space on the order of the Stage 2 output size. To use the older direct bucket scans instead:

```bash
uv run python ctg_reduction.py --stage stage3 --no-stage3-prebucket
```

For very large Stage 2 inputs, you can split the column filter into restartable shards that write multiple parquet outputs in the same Stage 2 directory:

```bash
uv run python ctg_reduction.py --stage stage2 --stage2-shard-count 3 --stage2-shard-index 0
uv run python ctg_reduction.py --stage stage2 --stage2-shard-count 3 --stage2-shard-index 1
uv run python ctg_reduction.py --stage stage2 --stage2-shard-count 3 --stage2-shard-index 2
```

These commands use `DEFAULT_STAGE2_DIR` from `config.py`. Rerunning the same sharded command skips shard files that already exist.

## Outputs

Main final outputs are written under `DEFAULT_STAGE7_DIR`:
- `registry.csv`: matched registry metadata, one row per `BabyID`
- `ctg_final.parquet`: anonymized CTG data linked by `BabyID`

Intermediate outputs for each stage are written under `DEFAULT_REDUCTION_ROOT`; the
pretraining export lives in `stage_3_sessionfilter/all_sessions/`.

## Checks

`uv run mypy`, `uv run pytest` (synthetic DuckDB/pyarrow fixtures, no patient data) and
`uvx ruff check .` from the repository root's `ruff.toml`.

## Notes

- Stage 7 supports SNQ input as `.xlsx`, `.xls`, or `.csv`.
- `match_loss_report.py` explains why registry births do not end up in the Stage 7 output. It reuses the Stage 7 registry cleaning and day-window rule from `registry_matching.py`, then assigns every registry birth row exactly one category (`registry_row_excluded` with sub-reasons `short_personnummer` / `missing_apgar5` / `missing_birth_day`, `no_ctg_for_patient`, `ctg_only_outside_window`, `dropped_stage4_duplicates`, `dropped_stage5_short_signal`, `multiple_ctg_matches`, `ctg_shared_by_multiple_registry_rows`, `matched`) by walking the Stage 3, Stage 4 and Stage 5.5 outputs. It prints markdown tables of counts and percentages, a histogram of the nearest Stage 3 CTG day offset for the outside-window rows (to judge whether the birth day / day-before window is too tight) and per-birth-year counts; only counts are printed, never PatientIDs or BabyIDs. Run it after Stage 5.5 with `uv run python match_loss_report.py` (defaults from `config.py`; override with `--registry-csv`, `--stage3`, `--stage4`, `--stage5-5`; `--out report.md` also writes the markdown to a file; `--no-progress` disables the DuckDB progress bar).
- Legacy experiments and old scripts are kept in `legacy/`.
- Local analysis utilities and generated artifacts are not part of the main pipeline.
