# Project Versions

## Current

### CTG3 multimodal
- Config: `configs/ctg3_multimodal.toml`
- Scripts:
  - `scripts/make_splits_multimodal.py`
  - `scripts/preprocess_multimodal.py`
  - `scripts/train_multimodal_tcn.py`
  - `scripts/run_multimodal_ablation_study.py`
  - `scripts/train_xgboost_registry.py`
  - `scripts/train_xgboost_tcn_embeddings.py`
  - Self-supervised encoder pretraining (optional, see README "Self-supervised pretraining"):
    - `scripts/preprocess_pretrain.py`
    - `scripts/pretrain_tcn.py`
    - `scripts/train_multimodal_tcn.py --init-encoder ... [--freeze-encoder-epochs N]`
- Artifacts: `artifacts_ctg3/` (pretraining shards + `encoder.pt` under `artifacts_ctg3/pretrain/`)
- Shared implementation:
  - `src/ctg_ml/multimodal_config.py` (incl. `[pretrain]` section, `train.init_encoder`)
  - `src/ctg_ml/multimodal_registry.py`
  - `src/ctg_ml/multimodal_preprocess.py`
  - `src/ctg_ml/pretrain_preprocess.py` (unlabeled window cutting, val/test BabyID exclusion)
  - `src/ctg_ml/pretrain.py` (masked reconstruction, encoder loading/freezing)
  - `src/ctg_ml/models.py` (`TCNEncoder.encode_sequence`, `MaskedReconstructionTCN`)

Use this for new experiments.

## Legacy

### CTG2 multimodal (removed 2026-09-03)
The CTG2 config, wrapper scripts (`make_splits_ctg2`, `preprocess_ctg2_multimodal`,
`train_ctg2_multimodal`, `run_ctg2_ablation_study`), the `ctg2_*` modules and
`docs/CTG2_MULTIMODAL_ARCHITECTURE.md` were deleted together with `CTG_preprocess/legacy/`.
They are preserved at the git tag `legacy-last-commit-2026-09-03`; existing `artifacts_ctg2/`
outputs on disk are unaffected.

### CTG1/simple binary
- Config: `configs/default.toml`
- Scripts:
  - `scripts/make_splits.py`
  - `scripts/train_baseline.py`
  - `scripts/preprocess_tcn.py`
  - `scripts/train_tcn.py`
- Artifacts: `artifacts/`

This is the original Apgar5 binary workflow.
