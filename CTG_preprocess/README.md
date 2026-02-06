CTG Preprocess

Overview
- Main pipeline scripts:
  - `ctg_reduction.py` for Stages 1–6 (+5.5)
  - `registry_matching.py` for Stage 7
- Optional utilities (sanity checks / plots) are in separate scripts.

Configuration
- Edit `config.py` for all paths and defaults.
- Key outputs:
  - Reduction root: `DEFAULT_REDUCTION_ROOT`
  - Stage 7 outputs: `DEFAULT_STAGE7_REGISTRY_CSV`, `DEFAULT_STAGE7_CTG_PARQUET`

Stages 1–6 (+5.5)
- Stage 1 (time filter):
  `uv run python ctg_reduction.py --stage stage1`
- Stage 2 (column filter):
  `uv run python ctg_reduction.py --stage stage2`
- Stage 3 (session filter):
  `uv run python ctg_reduction.py --stage stage3`
- Stage 4 (duplicate filter):
  `uv run python ctg_reduction.py --stage stage4`
- Stage 5 (quality filter):
  `uv run python ctg_reduction.py --stage stage5`
- Stage 5.5 (sort + add anchor date):
  `uv run python ctg_reduction.py --stage stage5_5`
- Stage 6 (partition by date):
  `uv run python ctg_reduction.py --stage stage6`

Stage 7 (Registry Matching)
- Matches registry data to CTG and writes final anonymized outputs:
  `uv run python registry_matching.py`
- Outputs:
  - `registry.csv` with BabyID, birth_day, apgar5
  - `ctg_final.parquet` with BabyID, Timestamp, FHR, toco

Plots (optional)
- Plot a random BabyID with apgar:
  `uv run python stage7_plot.py`
- Plot a random BabyID with a specific apgar score:
  `uv run python stage7_plot.py --apgar 9`

Sanity / Summary (optional)
- Stage summary across all outputs (fast by default):
  `uv run python stage_summary.py`

Notes on legacy scripts
- Earlier experiments (e.g. `main.py`, `partition_ctg.py`, older sanity scripts) are kept for reference.
- If you want a cleaner repo view, move them into a `legacy/` folder or add them to `.gitignore`.

Legacy
- Older experiments and scripts have been moved to `legacy/`.
- Generated plots moved to `legacy/plots/`.
