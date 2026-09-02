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

Edit `config.py` before running the pipeline.

Important paths:
- `DEFAULT_STAGE0_DIR`: directory with the raw CTG parquet files from the converter
- `DEFAULT_PATIENT_CSV`: main registry file (`gravniva.csv`)
- `DEFAULT_SNQ_FILE`: SNQ registry file
- `DEFAULT_REDUCTION_ROOT`: root directory for Stage 1-7 outputs

## Pipeline

Run the stages in order:

```bash
uv run python ctg_reduction.py --stage stage1
uv run python ctg_reduction.py --stage stage2
uv run python ctg_reduction.py --stage stage3
uv run python ctg_reduction.py --stage stage4
uv run python ctg_reduction.py --stage stage5
uv run python ctg_reduction.py --stage stage5_5
uv run python ctg_reduction.py --stage stage6
uv run python registry_matching.py
```

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

Intermediate outputs for each stage are written under `DEFAULT_REDUCTION_ROOT`.

## Notes

- Stage 7 supports SNQ input as `.xlsx`, `.xls`, or `.csv`.
- Legacy experiments and old scripts are kept in `legacy/`.
- Local analysis utilities and generated artifacts are not part of the main pipeline.
