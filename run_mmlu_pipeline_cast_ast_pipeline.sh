#!/usr/bin/env bash

set -euo pipefail

# ─── Models ───────────────────────────────────────────────────────────────────
MODELS=(
    "google/gemma-3-4b-it"
    "Qwen/Qwen2.5-3B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B-Instruct"
)

METHODS=("cast" "ast")

declare -A METHOD_CONFIG
METHOD_CONFIG["ast"]="benchmark_configs/ast.yaml"
METHOD_CONFIG["cast"]="benchmark_configs/cast.yaml"

UNSTEERED_CONFIG="benchmark_configs/unsteered.yaml"
MMLU_OUTPUT_DIR="./mmlu_steered_output"
MMLU_BATCH_SIZE=4

# ─── Parse flags ──────────────────────────────────────────────────────────────
MMLU_ONLY=false
SINGLE_MODEL=""
SINGLE_METHOD=""
SKIP_UNSTEERED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mmlu-only)      MMLU_ONLY=true; shift ;;
        --model)          SINGLE_MODEL="$2"; shift 2 ;;
        --method)         SINGLE_METHOD="$2"; shift 2 ;;
        --skip-unsteered) SKIP_UNSTEERED=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

[[ -n "$SINGLE_MODEL"  ]] && MODELS=("$SINGLE_MODEL")
[[ -n "$SINGLE_METHOD" ]] && METHODS=("$SINGLE_METHOD")

# ─── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Match your filename convention: "/" → "--"
safe_name() {
    echo "$1" | sed 's#/#--#g'
}

# ── Skip checks based on final output file ────────────────────────────────────
unsteered_done() {
    local model_safe
    model_safe=$(safe_name "$1")
    local f="$MMLU_OUTPUT_DIR/${model_safe}_unsteered_mmlu_5shot_logprob.json"
    [[ -f "$f" ]]
}

method_done() {
    local model_safe method
    model_safe=$(safe_name "$1")
    method="$2"
    local f="$MMLU_OUTPUT_DIR/${model_safe}_${method}_mmlu_5shot_logprob.json"
    [[ -f "$f" ]]
}

# ── Config generator (YAML-safe, strength only) ───────────────────────────────
make_config() {
    local model="$1"
    local method="$2"
    local base="${METHOD_CONFIG[$method]}"
    local tmp
    tmp=$(mktemp /tmp/"${method}"_XXXXXX.yaml)

    python3 - "$base" "$tmp" "$model" <<'PYEOF'
import sys, yaml

src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]

STRENGTH = {
    "Qwen/Qwen2.5-3B-Instruct":           2.22,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.25,
    "google/gemma-3-4b-it":               550,
    "google/gemma-3-12b-it":              550,
    "meta-llama/Llama-3.1-8B-Instruct":   0.4,
}

# Number of transformer layers per model
NUM_LAYERS = {
    "Qwen/Qwen2.5-3B-Instruct":           36,
    "google/gemma-3-4b-it":               32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
    "google/gemma-3-12b-it":              48,
    "meta-llama/Llama-3.1-8B-Instruct":   32,
}

# Layer range as fractions of total depth
LAYER_RANGE_START_FRAC = 0.15
LAYER_RANGE_END_FRAC   = 0.50

with open(src) as f:
    cfg = yaml.safe_load(f)

if model in STRENGTH:
    val = float(STRENGTH[model])
    cfg.setdefault("validation_selected", {})
    cfg["validation_selected"]["strength"] = {
        "default": val,
        "values": [val]
    }

if model in NUM_LAYERS:
    n = NUM_LAYERS[model]
    cfg.setdefault("validation_fixed", {})
    cfg["validation_fixed"]["layer_range_start"] = max(1, round(n * LAYER_RANGE_START_FRAC))
    cfg["validation_fixed"]["layer_range_end"]   = min(n - 1, round(n * LAYER_RANGE_END_FRAC))

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF

    echo "$tmp"
}

# ─── Main loop ────────────────────────────────────────────────────────────────
mkdir -p "$MMLU_OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
    log "========================================================"
    log "Model: $MODEL"
    log "========================================================"

    MODEL_SAFE=$(safe_name "$MODEL")

    # ── Unsteered ─────────────────────────────────────────────────────────────
    if [[ "$SKIP_UNSTEERED" == true ]]; then
        log "[unsteered] Skipped via --skip-unsteered"
    else
        if unsteered_done "$MODEL"; then
            log "[unsteered] Skipping (already done)"
        else
            log "[unsteered] Running MMLU"
            python -m src.patching.inference.logprob_mmlu_eval \
                --model      "$MODEL" \
                --config     "$UNSTEERED_CONFIG" \
                --output-dir "$MMLU_OUTPUT_DIR" \
                --batch-size "$MMLU_BATCH_SIZE"
        fi
    fi

    # ── Methods ───────────────────────────────────────────────────────────────
    for METHOD in "${METHODS[@]}"; do
        log "--------------------------------------------------------"
        log "Method: $METHOD"
        log "--------------------------------------------------------"

        if method_done "$MODEL" "$METHOD"; then
            log "[$METHOD] Skipping (already done)"
            continue
        fi

        CONFIG=$(make_config "$MODEL" "$METHOD")
        log "  Temp config: $CONFIG"

        if [[ "$MMLU_ONLY" == false ]]; then
            log "[1/3] Training"
            python -m src.patching.train.train_ast_cast \
                --model  "$MODEL" \
                --config "$CONFIG"

            log "[2/3] Validation"
            python -m src.patching.validation.run_validation \
                --model  "$MODEL" \
                --config "$CONFIG"

            log "[3/3] WildGuard labeling"
            python -m src.patching.validation.label_validation_with_wildguard \
                --model  "$MODEL" \
                --config "$CONFIG"
        else
            log "[--mmlu-only] Skipping train/validation"
        fi

        log "[4/4] MMLU steered"
        python -m src.patching.inference.logprob_mmlu_eval\
            --model      "$MODEL" \
            --config     "$CONFIG" \
            --output-dir "$MMLU_OUTPUT_DIR" \
            --batch-size "$MMLU_BATCH_SIZE"

        rm -f "$CONFIG"
        log "Done: $MODEL / $METHOD"
    done

    echo ""
done

# ─── Report ───────────────────────────────────────────────────────────────────
log "Generating MMLU comparison report..."
python -m src.patching.inference.mmlu_report \
    --output-dir "$MMLU_OUTPUT_DIR" \
    --models     "${MODELS[@]}" \
    --methods    "${METHODS[@]}"

log "Pipeline complete."