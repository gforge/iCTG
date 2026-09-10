#!/usr/bin/env bash
# Sequential experiment queue for the ~4 GB GPU budget (one training at a time):
#
#   A  base config + event features, random init, 3 seeds
#   B  base config, encoder from the existing pretrained encoder.pt, 3 seeds
#   C  base config, registry-only and CTG-only ablations, 3 seeds
#   D  variant without intervention targets, 3 seeds
#   E  variant with 180-minute history, 3 seeds (batch 32)
#
# Every run writes --metrics-out (+ per-sample test predictions); each block is recorded in
# the benchmark history and evaluated clinically on its first seed. Resume with
# START_BLOCK=<A..E>. Intended for tmux:  tmux new -s ctg-exp 'CTG_ML/scripts/run_experiments.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export CTG_DUCKDB_TEMP_DIR="${CTG_DUCKDB_TEMP_DIR:-/srv/data/input/iCTG/processed/duckdb_tmp}"
SEEDS=(${SEEDS:-51 52 53})
COHORT="${COHORT:-76855}"
BASE=configs/ctg3_multimodal.toml
NOINT=configs/ctg3_multimodal_no_interventions.toml
LONG=configs/ctg3_multimodal_180m.toml
START_BLOCK="${START_BLOCK:-A}"
LOG_DIR=artifacts_ctg3/logs
mkdir -p "$LOG_DIR"

log() { echo "$(date '+%F %T') $*"; }

train() {  # train <config> <run-name> <metrics-out> [extra args...]
    local cfg="$1" name="$2" metrics="$3"; shift 3
    if [ -f "$metrics" ]; then log "skip $name (metrics exist)"; return; fi
    log "train $name"
    uv run --no-sync python scripts/train_multimodal_tcn.py --config "$cfg" --run-name "$name" \
        --metrics-out "$metrics" --deterministic --no-progress "$@" \
        2>&1 | tee "$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log" | { grep -E "^Epoch|^TEST|Early stopping|Saved metrics|Error" || true; }
    [ -f "$metrics" ] || { log "FAILED $name (no metrics written)"; exit 1; }
}

record() {  # record <name> <notes> <metrics files...>
    local name="$1" notes="$2"; shift 2
    uv run --no-sync python scripts/record_benchmark.py --metrics "$@" --name "$name" --cohort "$COHORT" --notes "$notes"
}

clinical() {  # clinical <predictions.npz> <registry.csv> <out.md>
    uv run --no-sync python scripts/evaluate_clinical.py --predictions "$1" --registry "$2" --out "$3" | tail -20
}

preprocess() {  # preprocess <config> <artifacts dir>
    local cfg="$1" art="$2"
    if ls "$art"/tcn_multimodal_*/test.npz >/dev/null 2>&1; then log "skip preprocess for $cfg"; return; fi
    log "splits + preprocess for $cfg"
    uv run --no-sync python scripts/make_splits_multimodal.py --config "$cfg" | tail -5
    uv run --no-sync python scripts/preprocess_multimodal.py --config "$cfg" | tail -5
}

block_A() {
    # the base artifacts are rebuilt so the event features enter the tabular vector
    rm -f artifacts_ctg3/tcn_multimodal_60m/*.npz
    preprocess "$BASE" artifacts_ctg3
    local files=()
    for s in "${SEEDS[@]}"; do
        train "$BASE" "events_random_s$s" "artifacts_ctg3/metrics_events_random_s$s.json" --seed-override "$s"
        files+=("artifacts_ctg3/metrics_events_random_s$s.json")
    done
    record events_random_init "Base config + event features (clinician CTG classification, lactate, BP, note flags), composite-first monitoring, random init." "${files[@]}"
    clinical "artifacts_ctg3/metrics_events_random_s${SEEDS[0]}_predictions.npz" data/CTG3/registry.csv "artifacts_ctg3/clinical_events_random_s${SEEDS[0]}.md"
}

block_B() {
    local files=()
    for s in "${SEEDS[@]}"; do
        train "$BASE" "events_pretrained_s$s" "artifacts_ctg3/metrics_events_pretrained_s$s.json" \
            --seed-override "$s" --init-encoder artifacts_ctg3/pretrain/encoder.pt --freeze-encoder-epochs 2
        files+=("artifacts_ctg3/metrics_events_pretrained_s$s.json")
    done
    record events_pretrained_init "As events_random_init but encoder initialised from masked-reconstruction pretraining (frozen 2 epochs)." "${files[@]}"
}

block_C() {
    local reg=() ctg=()
    for s in "${SEEDS[@]}"; do
        train "$BASE" "events_registry_only_s$s" "artifacts_ctg3/metrics_events_registry_only_s$s.json" --seed-override "$s" --ablate-sequence
        train "$BASE" "events_ctg_only_s$s" "artifacts_ctg3/metrics_events_ctg_only_s$s.json" --seed-override "$s" --ablate-tabular
        reg+=("artifacts_ctg3/metrics_events_registry_only_s$s.json"); ctg+=("artifacts_ctg3/metrics_events_ctg_only_s$s.json")
    done
    record events_registry_only "Ablation: CTG input zeroed (registry + event features only)." "${reg[@]}"
    record events_ctg_only "Ablation: tabular input replaced by the train mean (CTG only)." "${ctg[@]}"
}

block_D() {
    preprocess "$NOINT" artifacts_ctg3_no_interventions
    local files=()
    for s in "${SEEDS[@]}"; do
        train "$NOINT" "noint_random_s$s" "artifacts_ctg3_no_interventions/metrics_noint_random_s$s.json" --seed-override "$s"
        files+=("artifacts_ctg3_no_interventions/metrics_noint_random_s$s.json")
    done
    record no_interventions_random_init "Variant B: emergency CS, fetal distress and delivery mode are not targets (interventions as description only)." "${files[@]}"
    clinical "artifacts_ctg3_no_interventions/metrics_noint_random_s${SEEDS[0]}_predictions.npz" data/CTG3/registry.csv "artifacts_ctg3_no_interventions/clinical_noint_random_s${SEEDS[0]}.md"
}

block_E() {
    preprocess "$LONG" artifacts_ctg3_180m
    local files=()
    for s in "${SEEDS[@]}"; do
        train "$LONG" "long180_random_s$s" "artifacts_ctg3_180m/metrics_long180_random_s$s.json" --seed-override "$s"
        files+=("artifacts_ctg3_180m/metrics_long180_random_s$s.json")
    done
    record history180_random_init "Variant C: last 180 minutes before the window end (stage 8b), batch 32, random init." "${files[@]}"
    clinical "artifacts_ctg3_180m/metrics_long180_random_s${SEEDS[0]}_predictions.npz" data/CTG3/registry.csv "artifacts_ctg3_180m/clinical_long180_random_s${SEEDS[0]}.md"
}

started=0
for block in A B C D E; do
    if [ "$block" = "$START_BLOCK" ]; then started=1; fi
    if [ "$started" -eq 1 ]; then log "==> block $block"; "block_$block"; fi
done
log "all blocks finished"
