#!/bin/bash
# =============================================================================
# Contrastive-decoding add-on for a model whose SFT baseline already exists.
# Trains the specialist (full fine-tune) then runs the lambda sweep + leakage.
# Used for Mistral-7B, whose specialist is a full-FT same-size model (no small
# same-tokenizer variant exists); the paged 8-bit optimizer keeps it within 96GB.
#
# Usage: sbatch slurm/contrastive_only.sbatch <MODEL_ID> <TAG> <SEED> <SPECIALIST_ID>
#   e.g. sbatch slurm/contrastive_only.sbatch mistralai/Mistral-7B-v0.3 mistral7b 42 mistralai/Mistral-7B-v0.3
# =============================================================================
#SBATCH --job-name=cs_contrast
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
[ -f scripts/train_sft.py ] || { echo "Run this from the CopyShield repo root"; exit 1; }
source .venv/bin/activate
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/shared/models/huggingface/hub
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

MODEL="${1:?MODEL_ID}"; TAG="${2:?TAG}"; SEED="${3:?SEED}"; SPECIALIST="${4:?SPECIALIST_ID}"
SFT=models/${TAG}_sft_s${SEED}
THETA=models/${TAG}_theta_s_s${SEED}
OUT=outputs/${TAG}_s${SEED}
THRESH=outputs/calibration/thresholds.json
LITERAL=data/eval_literal.json
NONLITERAL=data/eval_nonliteral_200.json
QA=data/qa_book_questions_200.json
mkdir -p "$OUT/contrastive" logs

echo "[contrastive] $TAG s$SEED  main_sft=$SFT  specialist=$SPECIALIST  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -e "$SFT/adapter_model.safetensors" ]; then
  echo "!! main SFT adapter missing ($SFT) — aborting (need the memorized model first)"; exit 1
fi

# Step 1: train specialist (full fine-tune, paged 8-bit optimizer fits 7B in 96GB)
if [ ! -e "$THETA/config.json" ]; then
  echo ">> train specialist"
  python scripts/contrastive.py train --corpus data/books_corpus.txt \
    --specialist_model "$SPECIALIST" --output_dir "$THETA" --epochs 20 --seed "$SEED" \
    || { echo "FAIL specialist train"; exit 1; }
else echo ">> SKIP specialist train (exists)"; fi

# Step 2: contrastive generation (lambda sweep)
if [ ! -e "$OUT/contrastive/qa_contrastive_lam2p0.json" ]; then
  echo ">> contrastive generate (lambda sweep)"
  python scripts/contrastive.py generate --base_model "$MODEL" --main_model "$SFT" \
    --specialist_dir "$THETA" --literal_data "$LITERAL" --nonliteral_data "$NONLITERAL" \
    --qa_data "$QA" --lambdas 2.0 8.0 --output_dir "$OUT/contrastive" \
    || { echo "FAIL contrastive generate"; exit 1; }
else echo ">> SKIP contrastive generate (exists)"; fi

# Step 3: leakage eval per lambda
for lam in 2p0 8p0; do
  for kind in literal nonliteral; do
    in="$OUT/contrastive/${kind}_contrastive_lam${lam}.json"
    out="$OUT/contrastive/leak_${kind}_lam${lam}.csv"
    if [ ! -f "$in" ]; then
      echo "!! missing leakage input: $in"; exit 1
    fi
    if [ -f "$out" ]; then
      echo ">> SKIP leakage (exists $out)"; continue
    fi
    python scripts/evaluate_leakage.py --input "$in" \
        --metrics all --thresholds "$THRESH" --output "$out" \
        || { echo "!! FAIL leakage $in"; exit 1; }
  done
done

date > "$OUT/CONTRASTIVE_DONE"
echo "[contrastive] DONE $TAG s$SEED $(date)"
