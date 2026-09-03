#!/usr/bin/env bash
# Run the full CTG reduction + registry matching pipeline (stages 1-8) sequentially,
# logging each stage under the reduction root. Intended to run inside tmux:
#
#   tmux new -s ctg-pipeline 'CTG_preprocess/run_pipeline.sh'
#
# Paths come from config.py (defaults: the shared server layout, override with the CTG_*
# env variables). Set START_STAGE to resume, e.g. START_STAGE=stage3 ./run_pipeline.sh
#
# Stage 3 also writes the all-sessions export used for self-supervised pretraining.
# Stage 8 time-shifts registry.csv, ctg_final.parquet and the all-sessions export into
# stage_8_timeshift/ (the deliverable); the secrets it needs are generated on first use
# under <reduction root>/secrets unless CTG_BABYID_SALT / CTG_TIMESHIFT_SECRET are set.

set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(uv run --no-sync python -c 'import config; print(config.DEFAULT_REDUCTION_ROOT)')"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
START_STAGE="${START_STAGE:-stage1}"
STAGES=(stage1 stage2 stage3 stage4 stage5 stage5_5 stage6 stage7 stage8 reports)

run_stage() {
    local stage="$1"
    local log="$LOG_DIR/${stage}_$(date +%Y%m%d_%H%M%S).log"
    echo "==> $stage  (log: $log)"
    case "$stage" in
        stage3)
            uv run --no-sync python ctg_reduction.py --stage stage3 --no-progress \
                --stage3-all-sessions-out 2>&1 | tee "$log" ;;
        stage7)
            uv run --no-sync python registry_matching.py --no-progress 2>&1 | tee "$log" ;;
        stage8)
            uv run --no-sync python time_shift.py 2>&1 | tee "$log" ;;
        reports)
            uv run --no-sync python cohort_report.py 2>&1 | tee "$log"
            uv run --no-sync python match_loss_report.py --no-progress \
                --out "$ROOT/match_loss_report.md" 2>&1 | tee -a "$log" ;;
        *)
            uv run --no-sync python ctg_reduction.py --stage "$stage" --no-progress 2>&1 | tee "$log" ;;
    esac
}

started=0
for stage in "${STAGES[@]}"; do
    if [ "$stage" = "$START_STAGE" ]; then started=1; fi
    if [ "$started" -eq 1 ]; then run_stage "$stage"; fi
done
echo "Pipeline finished. Outputs under $ROOT (deliverable: stage_8_timeshift/)"
