#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
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

MODEL="${1:?usage: sbatch sft_baseline.sbatch <hf_model_id> <out_name> [seed] [epochs]}"
OUT="${2:?provide out_name}"
SEED="${3:-42}"
EPOCHS="${4:-20}"
echo "[job $SLURM_JOB_ID] model=$MODEL out=$OUT seed=$SEED epochs=$EPOCHS host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/train_sft.py --model "$MODEL" --data data/books_corpus.txt \
    --output "models/$OUT" --seed "$SEED" --epochs "$EPOCHS"
echo "[job $SLURM_JOB_ID] DONE"
