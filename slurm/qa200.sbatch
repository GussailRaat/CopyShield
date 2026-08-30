#!/bin/bash
# =============================================================================
# QA-track augmentation from 50 to 200 examples.
# For one (base_model, tag, seed): regenerate ONLY the QA track on the new
# 200-question eval set for every method + the raw pre-SFT base, then judge with
# the same AI judge used elsewhere. Literal/non-literal are untouched.
#
# Usage: sbatch slurm/qa200.sbatch <BASE_MODEL> <TAG> <SEED>
#   sbatch slurm/qa200.sbatch meta-llama/Llama-3.1-8B mistral..  -> see launch below
# =============================================================================
#SBATCH --job-name=cs_qa200
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
[ -f scripts/train_sft.py ] || { echo "Run this from the CopyShield repo root"; exit 1; }
source .venv/bin/activate
[ -f slurm/env.sh ] && source slurm/env.sh
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/shared/models/huggingface/hub
unset TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM=false

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then echo "!! no ANTHROPIC_API_KEY"; exit 1; fi

MODEL="${1:?BASE_MODEL}"; TAG="${2:?TAG}"; SEED="${3:?SEED}"
SFT=models/${TAG}_sft_s${SEED}
DPO=models/${TAG}_dpo_s${SEED}
THETA=models/${TAG}_theta_s_s${SEED}
CLF=models/${TAG}_act_s${SEED}/classifiers
QA=data/qa_book_questions_200.json
OUT=outputs/${TAG}_s${SEED}/qa200
RAW=outputs/${TAG}_rawbase200
JUDGE="claude-sonnet-4-5-20250929"
mkdir -p "$OUT/contrastive" "$OUT/activation" "$RAW" logs

echo "[qa200] $TAG s$SEED  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- generation (QA only) ----
[ -f "$OUT/sft_qa.json" ] || python scripts/generate_outputs.py \
  --base_model "$MODEL" --model "$SFT" --qa "$QA" --output_qa "$OUT/sft_qa.json" \
  || { echo "FAIL sft gen"; exit 1; }

[ -f "$OUT/dpo_qa.json" ] || python scripts/generate_outputs.py \
  --base_model "$MODEL" --sft_adapter "$SFT" --model "$DPO" --qa "$QA" --output_qa "$OUT/dpo_qa.json" \
  || { echo "FAIL dpo gen"; exit 1; }

[ -f "$OUT/contrastive/qa_contrastive_lam8p0.json" ] || python scripts/contrastive.py generate \
  --base_model "$MODEL" --main_model "$SFT" --specialist_dir "$THETA" \
  --qa_data "$QA" --lambdas 2.0 8.0 --output_dir "$OUT/contrastive" \
  || { echo "FAIL contrastive gen"; exit 1; }

[ -f "$OUT/activation/qa_activation_thr0p5.json" ] || python scripts/activation.py generate \
  --base_model "$MODEL" --sft_adapter "$SFT" --classifier_dir "$CLF" \
  --qa_data "$QA" --threshold 0.5 --output_dir "$OUT/activation" \
  || { echo "FAIL activation gen"; exit 1; }

# raw pre-SFT base (seed-independent; guard by model tag)
[ -f "$RAW/qa_outputs.json" ] || python scripts/generate_outputs.py \
  --base_model "$MODEL" --no_adapter --qa "$QA" --output_qa "$RAW/qa_outputs.json" \
  || { echo "FAIL rawbase gen"; exit 1; }

# ---- judging (same judge model as all other utility evals) ----
judge() {
  local in="$1"; local out="$2"
  if [ ! -f "$in" ]; then
    echo "!! missing judge input: $in"; exit 1
  fi
  if [ -f "$out" ]; then
    echo ">> SKIP judge (exists $out)"; return 0
  fi
  python scripts/evaluate_utility.py --input "$in" --type qa --model "$JUDGE" --output "$out" --resume \
    || { echo "FAIL judge $in"; exit 1; }
}
judge "$OUT/sft_qa.json"                              "$OUT/sft_qa_utility.json"
judge "$OUT/dpo_qa.json"                              "$OUT/dpo_qa_utility.json"
judge "$OUT/contrastive/qa_contrastive_lam2p0.json"  "$OUT/contrastive/qa_lam2p0_utility.json"
judge "$OUT/contrastive/qa_contrastive_lam8p0.json"  "$OUT/contrastive/qa_lam8p0_utility.json"
judge "$OUT/activation/qa_activation_thr0p5.json"    "$OUT/activation/qa_utility.json"
judge "$RAW/qa_outputs.json"                          "$RAW/qa_utility.json"

date > "$OUT/QA200_DONE"
echo "[qa200] DONE $TAG s$SEED $(date)"
