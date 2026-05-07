#!/usr/bin/env bash
# Data creation pipeline: RedBench export → LLM inference → WildGuard → ELO
# Usage:
#   ./run_data_pipeline.sh        # full pipeline via SLURM
#   ./run_data_pipeline.sh --test # test mode: 1 small model, 10 samples (local, no SLURM)

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TEST_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_MODE=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Step 1: Export RedBench filtered data (local, fast) ───────────────────────
log "[1/4] Exporting RedBench filtered data..."
python -m src.data.export_redbench_filtered

# ── Test mode: run everything locally on one small model ──────────────────────
if [[ "$TEST_MODE" == true ]]; then
    log "Test mode: running locally on Qwen/Qwen2.5-3B-Instruct (10 samples)"
    python -m src.data.llm_inference --model "Qwen/Qwen2.5-3B-Instruct" --batch-size 4 --test 10
    python -m src.data.wildguard_label
    python -m src.data.elo_ranking --model_name "Qwen__Qwen2.5-3B-Instruct"
    log "Pipeline complete (test mode)."
    exit 0
fi

# ── Step 2: Submit LLM inference jobs via SLURM ───────────────────────────────
log "[2/4] Submitting LLM inference jobs..."
INFERENCE_JIDS=()
for slurm_file in slurm/llm_inference/*.slurm; do
    JID=$(sbatch --parsable "$slurm_file")
    INFERENCE_JIDS+=("$JID")
    log "  Submitted $(basename "$slurm_file") → job $JID"
done

if [[ ${#INFERENCE_JIDS[@]} -eq 0 ]]; then
    echo "No inference slurm scripts found in slurm/llm_inference/. Aborting." >&2
    exit 1
fi

INFERENCE_DEP="afterok:$(IFS=:; echo "${INFERENCE_JIDS[*]}")"

# ── Step 3: WildGuard labeling (depends on all inference jobs) ────────────────
log "[3/4] Submitting WildGuard labeling..."
WG_JID=$(sbatch --parsable --dependency="$INFERENCE_DEP" slurm/wildguard/wildguard_label.slurm)
log "  Submitted wildguard → job $WG_JID"

# ── Step 4: ELO ranking (depends on wildguard) ────────────────────────────────
log "[4/4] Submitting ELO ranking..."
ELO_JID=$(sbatch --parsable --dependency="afterok:${WG_JID}" slurm/elo/elo_ranking.slurm)
log "  Submitted elo → job $ELO_JID"

log "Pipeline submitted. Track with: squeue -u \$USER"
log "  Inference jobs : ${INFERENCE_JIDS[*]}"
log "  WildGuard job  : $WG_JID"
log "  ELO job        : $ELO_JID"