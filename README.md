# iCTG

Tools for turning raw cardiotocography (CTG) exports into an anonymized, registry-linked
research dataset and training outcome-prediction models on it.

The repository holds three independent Python projects, each with its own `pyproject.toml`,
lockfile and virtual environment:

| Directory | Purpose |
|-----------|---------|
| `src/ictg` (this level) | Stream raw JSON / zip exports into parquet (`ictg` CLI) |
| `CTG_preprocess/` | Seven-stage reduction of the parquet data and matching to registry data |
| `CTG_ML/` | TCN / XGBoost experiments predicting neonatal outcomes from the final dataset |

Each sub-directory has its own README with the pipeline details.

## Setup

All projects use [uv](https://github.com/astral-sh/uv) and Python >= 3.12.

```bash
# converter (this directory)
uv sync

# preprocessing
(cd CTG_preprocess && uv sync)

# ML experiments – see CTG_ML/README.md for how to reuse the shared /opt torch install
(cd CTG_ML && uv sync)
```

## Converter usage

```bash
# Preview the first 10 rows of a JSON export
uv run ictg "data/*.json" --preview 10

# Convert every export to parquet (one parquet file per input, resumable with --skip-existing)
uv run ictg "data/*.zip" --parquet-out output/ --skip-existing

# Long conversions: bin/convert.sh wraps the same command in a tmux session
bin/convert.sh "data/*.zip" --parquet-out output/
```

`uv run python -m ictg.convert.main` is equivalent to `uv run ictg`.

## Development checks

Lint and formatting use ruff with the shared `ruff.toml` at the repository root; type
checking uses mypy and tests use pytest, configured per project in each `pyproject.toml`.

```bash
bin/check.sh          # ruff + mypy + pytest for all three projects
bin/check.sh lint     # or: types | test
```

Inside a single project the same checks are `uvx ruff check .`, `uv run mypy` and
`uv run pytest`.
