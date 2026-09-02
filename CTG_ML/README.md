# CTG_ML

Machine learning experiments for predicting neonatal outcome (`apgar5`) from CTG time series.

## Goal (Phase 1)

Start with a binary classifier:

- `healthy` = `apgar5 >= 7`
- `at_risk` = `apgar5 <= 6`

The dataset is highly imbalanced (~1.2% `at_risk`), so evaluation should focus on:

- PR-AUC (average precision)
- ROC-AUC
- Recall / sensitivity at clinically reasonable thresholds
- Confusion matrix on a held-out test set

## Recommended progression

1. `BabyID`-level split (train / val / test, stratified on label)
2. Fast baseline on aggregated features (DuckDB + scikit-learn)
3. Sequence preprocessing (fixed-length tensors per pregnancy)
4. PyTorch TCN training

This repository scaffold implements steps 1-2 and includes a PyTorch TCN model skeleton for step 4.

## Setup with uv

`torch` is deliberately kept out of the default dependencies (it is in the optional
`torch` dependency group) because the CUDA wheels are several gigabytes.

On the shared GPU machine, reuse the torch build from the `/opt/compute` micromamba
environment by creating the venv on that interpreter with access to its site-packages:

```bash
uv venv --python /opt/compute/mamba-root/envs/ndl/bin/python3.13 --system-site-packages
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On any other machine install torch into the venv instead:

```bash
uv sync --group torch
```

Checks: `uv run mypy`, `uv run pytest`, and `uvx ruff check .` (configured by the
`ruff.toml` at the repository root).

## Current CTG3 Workflow

CTG3 is the current active multimodal version. It keeps the TCN sequence encoder, adds registry/tabular inputs after the pooled CTG embedding, and predicts multiple outputs at once.

Default config:

- `configs/ctg3_multimodal.toml`

The public config uses local placeholder paths under `data/`. Place the CTG and registry
files there, create symlinks, or edit the `[paths]` section before running:

- `data/CTG3/ctg_final.parquet`
- `data/CTG3/registry.csv`

Workflow:

```bash
uv run python scripts/make_splits_multimodal.py --config configs/ctg3_multimodal.toml
uv run python scripts/preprocess_multimodal.py --config configs/ctg3_multimodal.toml
uv run python scripts/train_multimodal_tcn.py --config configs/ctg3_multimodal.toml
```

Registry-only XGBoost baseline and feature-importance run:

```bash
uv run python scripts/train_xgboost_registry.py --config configs/ctg3_multimodal.toml
```

XGBoost on frozen TCN embeddings plus registry features:

```bash
uv run python scripts/train_xgboost_tcn_embeddings.py --config configs/ctg3_multimodal.toml
```

Design notes:

- CTG inputs: `FHR`, `toco`, one-hot `Hr1_SignalQuality` channels, and `padding_mask`
- Registry inputs: numeric/boolean/categorical columns encoded into a dense tabular vector
- Outputs: Apgar class heads (`0-10`), continuous pH heads, and binary heads for selected neonatal outcomes
- CTG3 adds `gestational_days`, `previous_c_section`, and `neonatal_anemia`
- The intended prediction moment is the last hour before birth, so late-labour variables in the config are intentional inputs

## Self-supervised pretraining

The TCN sequence encoder can be pretrained with masked reconstruction on unlabeled CTG
windows from *all* sessions of *all* pregnancies (antenatal and earlier labour sessions),
then used to initialise the supervised CTG3 model. Input is the stage 3 side output of
`CTG_preprocess`: either a single parquet file or a directory of `all_sessions_bucket_*.parquet`
files with columns `BabyID`, `session_id`, `Timestamp`, `FHR` (0 = missing), `toco`,
`Hr1_SignalQuality`, `in_final_window`. Settings live in the `[pretrain]` section of the config.

```bash
# 1) Cut 60-min windows (stride 30 min) from every (BabyID, session_id) into NPZ shards.
#    Val/test BabyIDs from artifacts_dir/splits.csv are ALWAYS excluded (fails without splits.csv
#    unless --allow-no-splits); windows overlapping the supervised last-hour window are dropped.
uv run python scripts/preprocess_pretrain.py --config configs/ctg3_multimodal.toml

# 2) Masked-reconstruction pretraining -> artifacts_ctg3/pretrain/encoder.pt (+ pretrain_metrics.json)
uv run python scripts/pretrain_tcn.py --config configs/ctg3_multimodal.toml

# 3) Supervised training initialised from the pretrained encoder (optionally frozen for N epochs)
uv run python scripts/train_multimodal_tcn.py --config configs/ctg3_multimodal.toml \
  --init-encoder artifacts_ctg3/pretrain/encoder.pt --freeze-encoder-epochs 2
```

Design notes:

- Windows are built with the same `_finalize_sequence` and channel order as the supervised
  pipeline (`FHR`, `toco`, one-hot `Hr1_SignalQuality`, `padding_mask`), so
  `sequence_encoder_state_dict` in `encoder.pt` loads strictly into
  `MultimodalMultitaskTCN.sequence_encoder`. Channel count and names are verified on load.
- Rows are reindexed onto a contiguous 1 Hz grid per session; a missing second becomes FHR=0
  (missing), like the parquet's own missing encoding. Windows need `min_signal_fraction`
  non-missing FHR samples.
- Masking zeroes random contiguous spans (`mask_span_seconds`, ~`mask_ratio` of steps) in the
  FHR/toco channels only. No extra mask-indicator channel is added because it would change the
  encoder's input width; in normalized space a zero is exactly what the supervised model sees for
  missing samples. The loss is MSE at masked positions where the raw signal was present.
- Pretraining windows are split 90/10 by BabyID for early stopping on reconstruction loss.
  `encoder.pt` also stores the pretraining normalization stats; the supervised run keeps using
  its own train-set stats (both are printed).
- `train.init_encoder` / `train.freeze_encoder_epochs` in the config are the defaults for the
  two CLI flags.

## Legacy Workflows

These are kept so earlier results can still be inspected or reproduced, but new work should start from the CTG3 workflow above.

- CTG1/simple binary workflow: `configs/default.toml`, `scripts/make_splits.py`, `scripts/preprocess_tcn.py`, `scripts/train_tcn.py`
- CTG2 multimodal workflow: `configs/ctg2_multimodal.toml`, `scripts/make_splits_ctg2.py`, `scripts/preprocess_ctg2_multimodal.py`, `scripts/train_ctg2_multimodal.py`
- CTG2 ablation tooling: `scripts/run_ctg2_ablation_study.py`
- Canonical shared implementation for new work: `src/ctg_ml/multimodal_config.py`, `src/ctg_ml/multimodal_registry.py`, `src/ctg_ml/multimodal_preprocess.py`
- Version map: `docs/PROJECT_VERSIONS.md`

## Notes

- Splits are created on `BabyID`, so no pregnancy leaks across train/val/test.
- The baseline is a sanity check and usually catches data issues early (join problems, leakage, label bugs).
- CTG3 preprocessing defaults to the last 60 minutes at 1 Hz (3600 steps).
