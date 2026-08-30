#!/bin/bash
# =============================================================================
# Raw-base Mistral-7B-v0.3 QA utility ceiling (pre-SFT, no defense).
# Generates 200 QA answers from the raw base model, then judges them with the
# SAME judge used for every other utility eval, so the ceiling is comparable.
# The mean QA utility here is the "% of ceiling" denominator for the Mistral
# results table (mirrors the LLaMA raw-base ceiling of 1.56 in the paper).
# =============================================================================
#SBATCH --job-name=cs_mistral_ceiling
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "!! ANTHROPIC_API_KEY empty — fill slurm/env.sh. Aborting."; exit 1
fi

MODEL=mistralai/Mistral-7B-v0.3
OUT=outputs/mistral7b_rawbase200
QA=data/qa_book_questions_200.json
JUDGE="claude-sonnet-4-5-20250929"
mkdir -p "$OUT" logs

echo "[ceiling] start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Step 1: generate QA answers from the RAW base model (no adapter, greedy, 200 tok)
if [ ! -f "$OUT/qa_outputs.json" ]; then
  echo ">> generate raw-base QA"
  python scripts/generate_outputs.py --base_model "$MODEL" --no_adapter \
    --qa "$QA" --max_new_tokens 200 \
    --output_qa "$OUT/qa_outputs.json" || { echo "FAIL generate"; exit 1; }
else echo ">> SKIP generate (exists)"; fi

# Step 2: judge with the SAME model used for all other utility evals
if [ ! -f "$OUT/qa_utility.json" ]; then
  echo ">> judge raw-base QA"
  python scripts/evaluate_utility.py --input "$OUT/qa_outputs.json" --type qa \
    --model "$JUDGE" --output "$OUT/qa_utility.json" --resume \
    || { echo "FAIL judge"; exit 1; }
else echo ">> SKIP judge (exists)"; fi

# Step 3: print the ceiling (mean QA utility)
python - <<'PY'
import json
r=json.load(open("outputs/mistral7b_rawbase200/qa_utility.json"))
qa=[x for x in r if x.get('sample_type')=='qa']
u=[x['utility_score'] for x in qa if x.get('utility_score',0)>0]
print(f"\n===== Mistral-7B-v0.3 RAW-BASE QA ceiling =====")
print(f"n={len(qa)}  mean QA utility (ceiling) = {sum(u)/len(u):.4f}")
PY

date > "$OUT/CEILING_DONE"
echo "[ceiling] COMPLETE $(date)"
