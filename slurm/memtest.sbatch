#!/bin/bash
# =============================================================================
# Memorization-recipe test: train a STRONGER-memorization SFT baseline and
# measure FREE-RUNNING (greedy) literal nv_recall, to see whether the recipe
# reaches Llama-like verbatim leakage (~0.26) before committing to a full
# pipeline re-run. Trains to models/${TAG}_sftDEEP_s${SEED}, generates literal
# only, runs leakage eval, prints the verdict.
#
# Usage:
#   sbatch slurm/memtest.sbatch <MODEL_ID> <TAG> <SEED> <EPOCHS> <LR>
#   e.g.  sbatch slurm/memtest.sbatch Qwen/Qwen3-8B-Base qwen3_8b 42 60 3e-4
# =============================================================================
#SBATCH --job-name=cs_memtest
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
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

MODEL="${1:?MODEL_ID}"; TAG="${2:?TAG}"; SEED="${3:?SEED}"; EPOCHS="${4:-60}"; LR="${5:-3e-4}"
SFT=models/${TAG}_sftDEEP_s${SEED}
OUT=outputs/${TAG}_sftDEEP_s${SEED}/sft_baseline
mkdir -p "$OUT" logs

echo "[memtest] $TAG s$SEED epochs=$EPOCHS lr=$LR  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -e "$SFT/adapter_model.safetensors" ]; then
  echo ">> train DEEP SFT"
  python scripts/train_sft.py --model "$MODEL" --data data/books_corpus.txt \
    --output "$SFT" --seed "$SEED" --epochs "$EPOCHS" --lr "$LR" || { echo "FAIL train"; exit 1; }
fi

echo ">> generate literal (free-running greedy)"
python scripts/generate_outputs.py --base_model "$MODEL" --model "$SFT" \
  --literal data/eval_literal.json \
  --output_literal "$OUT/literal_outputs.json" || { echo "FAIL gen"; exit 1; }

echo ">> leakage eval"
python scripts/evaluate_leakage.py --input "$OUT/literal_outputs.json" --metrics all \
  --thresholds outputs/calibration/thresholds.json --output "$OUT/leak_literal.csv" \
  || { echo "FAIL leakage eval"; exit 1; }

echo "=================== VERDICT ==================="
python - <<PY
import json
lk=json.load(open("$OUT/leak_literal.json"))
nv=[r.get('lcs_nv_recall',0) for r in lk]; rl=[r.get('rouge_l',0) for r in lk]; n=len(nv)
print(f"$TAG DEEP (epochs=$EPOCHS lr=$LR)  n={n}")
print(f"  nv_recall mean = {sum(nv)/n:.3f}   (target ~0.26 = Llama-like)")
print(f"  rougeL   mean = {sum(rl)/n:.3f}")
print(f"  verbatim (rougeL=1.0): {sum(1 for v in rl if v>0.999)}   strong (nv>0.2): {sum(1 for v in nv if v>0.2)} / {n}")
PY
echo "=============================================="
echo "[memtest] DONE $(date)"
