#!/usr/bin/env bash
# Point CTG_ML/data/CTG3/ at the stage 8 (time-shifted) deliverable of CTG_preprocess.
#
#   CTG_ML/scripts/link_stage8_data.sh            # uses config.DEFAULT_STAGE8_DIR
#   CTG_ML/scripts/link_stage8_data.sh /path/to/stage_8_timeshift
#
# Creates symlinks for the four inputs configs/ctg3_multimodal.toml expects and refuses to
# proceed if any of them is missing, so a training run never silently uses stale data.
set -euo pipefail

ML_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PREPROCESS_DIR="$ML_DIR/../CTG_preprocess"
STAGE8="${1:-$(cd "$PREPROCESS_DIR" && uv run --no-sync python -c 'import config; print(config.DEFAULT_STAGE8_DIR)')}"
DATA_DIR="$ML_DIR/data/CTG3"
mkdir -p "$DATA_DIR"

link() {  # link <source under STAGE8> <target name under DATA_DIR>
    local src="$STAGE8/$1" dst="$DATA_DIR/$2"
    if [ ! -e "$src" ]; then
        echo "ERROR: missing $src (run stage 8 first: START_STAGE=stage8 CTG_preprocess/run_pipeline.sh)" >&2
        exit 1
    fi
    ln -sfn "$src" "$dst"
    echo "$dst -> $src"
}

link registry.csv registry.csv
link ctg_final.parquet ctg_final.parquet
link mothers.csv mothers.csv
link all_sessions ctg_pretrain.parquet   # directory of bucket files; the config accepts a directory

if [ -e "$STAGE8/timeshift_key.parquet" ]; then
    echo "NOTE: $STAGE8/timeshift_key.parquet re-identifies the dates; it is deliberately not linked."
fi
