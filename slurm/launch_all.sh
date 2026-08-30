#!/bin/bash
# =============================================================================
# Submit the full CopyShield experiment matrix.
# Each (model,seed) runs the complete pipeline as one GPU job; a utility-eval
# job is chained after it (afterok) so AI-judge scores run automatically once
# the GPU work + your ANTHROPIC_API_KEY are in place.
#
#   ./slurm/launch_all.sh          # Qwen matrix (default)
#   ./slurm/launch_all.sh llama    # Llama only (needs HF_TOKEN in slurm/env.sh)
#   ./slurm/launch_all.sh all      # Qwen + Llama
#
# Matrix:
#   Multi-model : qwen3_4b/8b/14b @ seed 42
#   Multi-seed  : qwen3_8b @ seeds 42,123,7   (stability run)
#   Llama redo  : llama3_8b @ seed 42          (gated -> needs HF token)
# =============================================================================
set -euo pipefail
[ -f scripts/train_sft.py ] || { echo "Run this from the CopyShield repo root"; exit 1; }
mkdir -p logs
[ -f slurm/env.sh ] && source slurm/env.sh

WHICH="${1:-qwen}"
QSPEC="Qwen/Qwen3-1.7B-Base"      # shared Qwen specialist (same tokenizer as 4B/8B/14B)
LSPEC="meta-llama/Llama-3.2-1B"   # Llama specialist

# submit <MODEL> <TAG> <SEED> <SPECIALIST>
submit() {
  local model="$1" tag="$2" seed="$3" spec="$4"
  local jid
  jid=$(sbatch --parsable --job-name="cs_${tag}_s${seed}" \
        slurm/pipeline.sbatch "$model" "$tag" "$seed" "$spec")
  echo "  pipeline $tag s$seed -> job $jid"
  # chain utility eval to run when the GPU pipeline succeeds
  local ujid
  ujid=$(sbatch --parsable --job-name="cs_util_${tag}_s${seed}" \
         --dependency=afterok:"$jid" slurm/utility_eval.sbatch "$tag" "$seed")
  echo "    utility $tag s$seed -> job $ujid (after $jid)"
}

echo "== Submitting matrix ($WHICH) =="
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "  [note] ANTHROPIC_API_KEY empty in slurm/env.sh — utility jobs are queued"
  echo "         (afterok) and will run once you add the key before pipelines finish."
fi

if [ "$WHICH" = "qwen" ] || [ "$WHICH" = "all" ]; then
  submit Qwen/Qwen3-4B-Base  qwen3_4b  42  "$QSPEC"
  submit Qwen/Qwen3-8B-Base  qwen3_8b  42  "$QSPEC"
  submit Qwen/Qwen3-8B-Base  qwen3_8b  123 "$QSPEC"
  submit Qwen/Qwen3-8B-Base  qwen3_8b  7   "$QSPEC"
  submit Qwen/Qwen3-14B-Base qwen3_14b 42  "$QSPEC"
fi

if [ "$WHICH" = "llama" ] || [ "$WHICH" = "all" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "  [skip] Llama: HF_TOKEN empty in slurm/env.sh (gated models). Add it, then:"
    echo "         ./slurm/launch_all.sh llama"
  else
    submit meta-llama/Llama-3.1-8B llama3_8b 42 "$LSPEC"
  fi
fi

echo "== Done. Monitor with: squeue -u \$USER =="
