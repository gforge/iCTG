# iCTG — agent instructions

## Patient data: never look at it, never send it anywhere

The data under `/srv/data/input/iCTG` (raw JSON/zip, parquet, `registry/*.csv|xlsx`,
`processed/**`) and anything derived from it is identifiable Swedish health data. These rules
apply to every agent and subagent, in every session, with no exceptions for "just one row",
"just to debug", or "it's already pseudonymised":

- **Never read, print, sample or display row-level content.** No `head`, `SELECT *`,
  `LIMIT n`, `.head()`, `.sample()`, `describe()` on free-text/ID columns, no `cat` of CSV/JSON,
  no opening parquet row groups to "see what it looks like". This includes PatientID, BabyID,
  personnummer, glopnr, timestamps of individual recordings, and free-text fields.
- **Never send it anywhere.** No copying to another host, no uploads, no pasting into
  artifacts, feedback, PR descriptions, commit messages, test fixtures, logs or memory files.
  Report outputs must contain only counts and aggregates.
- **Work from shape, not content.** Use parquet/CSV schema and column names, row counts,
  parquet metadata statistics, `COUNT(*)`, `COUNT(DISTINCT ...)`, histograms and other
  aggregates with at least 5 units per cell.
- **Compare with equality, don't inspect.** To check whether two things match (an ID in one
  table exists in another, a value equals an expected constant, two exports overlap), write a
  join / `==` / `IN` / `EXISTS` and report *how many* matched, never *which* ones.
- **Test on synthetic data.** All tests use generated DuckDB/pyarrow fixtures under
  `*/tests/`; never derive a fixture from real rows.
- If a task cannot be done without looking at real content, stop and say so instead of
  looking.

These rules are enforced by a PreToolUse hook (`.claude/settings.json` → `bin/guard_patient_data.py`)
that denies row-printing, row-returning and data-transfer commands touching the data roots. If it
blocks a legitimate aggregate query, rewrite the query (COUNT/GROUP BY without row output) rather
than bypassing the hook.

The `BabyID` salt in `CTG_preprocess/config.py` is a secret; do not paste it into messages.

## Project layout

- `src/ictg` — raw JSON/zip → parquet converter (`ictg` CLI, `bin/convert.sh`).
- `CTG_preprocess` — stages 1–7 (DuckDB/pyarrow reduction, registry matching, `run_pipeline.sh`).
- `CTG_ML` — TCN multimodal model, XGBoost baseline, self-supervised pretraining.

Each is its own `uv` project (Python ≥ 3.12). `CTG_ML` uses the shared torch env in
`/opt/compute/mamba-root/envs/ndl` via `--system-site-packages`; do not install torch into the venv.

## Checks before committing

`bin/check.sh` runs ruff (root `ruff.toml`), mypy and pytest for all three projects. Keep it green.
