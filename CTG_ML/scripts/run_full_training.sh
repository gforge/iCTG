#!/usr/bin/env bash
# Full CTG3 training chain on the current deliverable, sequentially, with one log per step
# under <artifacts_dir>/logs. Intended for tmux:
#
#   tmux new -s ctg-ml 'CTG_ML/scripts/run_full_training.sh'
#
# Steps (set START_STEP to resume, e.g. START_STEP=pretrain_preprocess):
#   splits               mother-level train/val/test split
#   preprocess           supervised last-hour windows
#   xgboost              registry-only baseline
#   train_random         multimodal TCN, random init
#   pretrain_preprocess  unlabeled windows from all sessions (val/test mothers excluded)
#   pretrain             masked-reconstruction pretraining of the encoder
#   train_pretrained     multimodal TCN initialised from the pretrained encoder
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/ctg3_multimodal.toml}"
ART="$(uv run --no-sync python -c "from ctg_ml.multimodal_config import load_multimodal_config as l; print(l('$CONFIG').paths.artifacts_dir)")"
LOG_DIR="$ART/logs"
mkdir -p "$LOG_DIR"
START_STEP="${START_STEP:-splits}"
STEPS=(splits preprocess xgboost train_random pretrain_preprocess pretrain train_pretrained)

run_step() {
    local step="$1"
    local log="$LOG_DIR/${step}_$(date +%Y%m%d_%H%M%S).log"
    echo "==> $step  (log: $log)"
    case "$step" in
        splits)
            uv run --no-sync python scripts/make_splits_multimodal.py --config "$CONFIG" 2>&1 | tee "$log" ;;
        preprocess)
            uv run --no-sync python scripts/preprocess_multimodal.py --config "$CONFIG" 2>&1 | tee "$log" ;;
        xgboost)
            uv run --no-sync python scripts/train_xgboost_registry.py --config "$CONFIG" 2>&1 | tee "$log" ;;
        train_random)
            uv run --no-sync python scripts/train_multimodal_tcn.py --config "$CONFIG" \
                --run-name random_init --no-progress 2>&1 | tee "$log" ;;
        pretrain_preprocess)
            uv run --no-sync python scripts/preprocess_pretrain.py --config "$CONFIG" --no-progress 2>&1 | tee "$log" ;;
        pretrain)
            uv run --no-sync python scripts/pretrain_tcn.py --config "$CONFIG" --no-progress 2>&1 | tee "$log" ;;
        train_pretrained)
            uv run --no-sync python scripts/train_multimodal_tcn.py --config "$CONFIG" \
                --init-encoder "$ART/pretrain/encoder.pt" --freeze-encoder-epochs 2 \
                --run-name pretrained_init --no-progress 2>&1 | tee "$log" ;;
    esac
}

started=0
for step in "${STEPS[@]}"; do
    if [ "$step" = "$START_STEP" ]; then started=1; fi
    if [ "$started" -eq 1 ]; then run_step "$step"; fi
done
echo "Training chain finished. Artifacts under $ART"
