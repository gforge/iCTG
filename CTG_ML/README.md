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

## Configuration

Edit `configs/default.toml` if needed. The default paths are wired to your current dataset location.

## Run

Create splits:

```bash
uv run python scripts/make_splits.py
```

Train a baseline (aggregated features, class-weighted logistic regression):

```bash
uv run python scripts/train_baseline.py
```

TCN skeleton (requires sequence tensors you preprocess first):

```bash
uv run python scripts/preprocess_tcn.py
uv run python scripts/train_tcn.py
```

## Notes

- Splits are created on `BabyID`, so no pregnancy leaks across train/val/test.
- The baseline is a sanity check and usually catches data issues early (join problems, leakage, label bugs).
- TCN preprocessing defaults to the last 20 minutes at 1 Hz (1200 steps) and adds an `fhr_missing_mask` channel.
- You can later switch to 60 minutes by editing `configs/default.toml` (`[sequence].window_minutes = 60`) and rerunning preprocessing.
