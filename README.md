# iCTG

A Python project for analyzing CTG (Cardiotocography) data

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

1. Install uv if you haven't already:

   ```bash
   # Using snap (Linux)
   sudo snap install astral-uv --classic
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

After installing dependencies with `uv sync`, you can run the script in several ways:

### Option 1: Using uv run (recommended)

```bash
uv run python -m ictg.convert_raw_ctg_to_dataframe --help
```

### Option 2: Install in editable mode and run as command

```bash
pip install -e .
ictg --help
```

### Option 3: Run directly

```bash
python src/ictg/convert_raw_ctg_to_dataframe.py --help
```

### Examples

```bash
# Preview first 10 rows from JSON files
uv run python -m ictg.convert_raw_ctg_to_dataframe "data/*.json" --preview 10

# Convert to Parquet
uv run python -m ictg.convert_raw_ctg_to_dataframe "data/*.zip" --parquet-out output/
```

## Dependencies

- pandas: For data manipulation
- pydantic: For data validation and models
