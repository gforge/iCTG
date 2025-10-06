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

Run the conversion script:

```bash
python convert_raw_CTG_to_dataframe.py --input data/Export_2025-05-22\ 10_14_42.json --output output.csv
```

Use `--help` for more options.

## Dependencies

- pandas: For data manipulation
- pydantic: For data validation and models
