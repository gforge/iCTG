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

```bash
uv venv
source .venv/bin/activate
uv sync
```

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
