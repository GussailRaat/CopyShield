#!/bin/bash
# =============================================================================
# CopyShield utility (AI-judge) evaluation for ONE (tag, seed).
# CPU + Anthropic API only — NO GPU. Consumes the *_outputs.json produced by
# pipeline.sbatch and writes utility_*.json + *_summary.csv next to them.
# Also emits refusal and degeneracy rates.
#
# Requires ANTHROPIC_API_KEY (from slurm/env.sh).
#
# Usage:
#   sbatch slurm/utility_eval.sbatch <TAG> <SEED>
#   e.g.  sbatch slurm/utility_eval.sbatch qwen3_8b 42
# =============================================================================
#SBATCH --job-name=cs_util
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
[ -f scripts/train_sft.py ] || { echo "Run this from the CopyShield repo root"; exit 1; }
source .venv/bin/activate
[ -f slurm/env.sh ] && source slurm/env.sh

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "!! ANTHROPIC_API_KEY is empty — fill slurm/env.sh first. Aborting."; exit 1
fi

TAG="${1:?usage: sbatch utility_eval.sbatch <TAG> <SEED>}"
SEED="${2:?provide SEED}"
OUT=outputs/${TAG}_s${SEED}
# Current pinned Sonnet-class judge, applied consistently across every eval.
JUDGE="claude-sonnet-4-5-20250929"

echo "[util $TAG s$SEED] start $(date)"

# util <nonliteral.json> <qa.json> <out-prefix>
util() {
  local nl="$1"; local qa="$2"; local prefix="$3"
  local inputs=(); local types=()
  [ -f "$nl" ] && { inputs+=("$nl"); types+=(nonliteral); }
  [ -f "$qa" ] && { inputs+=("$qa"); types+=(qa); }
  [ -f "$nl" ] || { echo "!! missing utility input: $nl"; exit 1; }
  [ -f "$qa" ] || { echo "!! missing utility input: $qa"; exit 1; }
  echo ">> utility: $prefix  <- ${inputs[*]}"
  python scripts/evaluate_utility.py --input "${inputs[@]}" --type "${types[@]}" \
      --model "$JUDGE" --output "${prefix}_utility.json" --resume \
      || { echo "!! FAIL utility $prefix"; exit 1; }
}

util "$OUT/sft_baseline/nonliteral_outputs.json" "$OUT/sft_baseline/qa_outputs.json" "$OUT/sft_baseline/sft"
util "$OUT/dpo/nonliteral_outputs.json"          "$OUT/dpo/qa_outputs.json"          "$OUT/dpo/dpo"
util "$OUT/activation/nonliteral_activation_thr0p5.json" "$OUT/activation/qa_activation_thr0p5.json" "$OUT/activation/act"
for lam in 2p0 4p0 8p0; do
  util "$OUT/contrastive/nonliteral_contrastive_lam${lam}.json" \
       "$OUT/contrastive/qa_contrastive_lam${lam}.json" \
       "$OUT/contrastive/contrastive_lam${lam}"
done

date > "$OUT/UTILITY_DONE"
echo "[util $TAG s$SEED] COMPLETE $(date)"
