#!/bin/bash
# =============================================================================
# CopyShield full pipeline for ONE (model, seed).
# Runs the whole DAG sequentially on a single GPU, idempotently:
#   SFT baseline -> generate+eval
#   Contrastive  -> train specialist -> generate (lambda sweep) + eval
#   DPO          -> build pairs -> train -> generate + eval
#   Activation   -> collect -> classify -> generate + eval
# Utility (AI-judge) eval is a separate job (slurm/utility_eval.sbatch) because
# it needs the Anthropic API, not a GPU.
#
# Every step is guarded by an output-exists check, so re-submitting a killed
# job resumes where it stopped.
#
# Usage:
#   sbatch slurm/pipeline.sbatch <MODEL_ID> <TAG> <SEED> <SPECIALIST_ID> ["<LAYERS>"]
# e.g.:
#   sbatch slurm/pipeline.sbatch Qwen/Qwen3-8B-Base qwen3_8b 42 Qwen/Qwen3-1.7B-Base
#   sbatch slurm/pipeline.sbatch meta-llama/Llama-3.1-8B llama3_8b 42 meta-llama/Llama-3.2-1B
# =============================================================================
#SBATCH --job-name=cs_pipe
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
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
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
# Secrets (HF token for gated Llama); safe no-op if empty.
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL="${1:?usage: sbatch pipeline.sbatch <MODEL_ID> <TAG> <SEED> <SPECIALIST_ID> [LAYERS]}"
TAG="${2:?provide TAG (e.g. qwen3_8b)}"
SEED="${3:?provide SEED}"
SPECIALIST="${4:?provide SPECIALIST_ID}"
LAYERS="${5:-8 12 16 20 24 28}"

EPOCHS_SFT=20
THRESH=outputs/calibration/thresholds.json
LITERAL=data/eval_literal.json
NONLITERAL=data/eval_nonliteral_200.json
QA=data/qa_book_questions_200.json

SFT=models/${TAG}_sft_s${SEED}
THETA=models/${TAG}_theta_s_s${SEED}
DPO=models/${TAG}_dpo_s${SEED}
DPO_PAIRS=data/dpo_pairs_${TAG}_s${SEED}.json
ACT=models/${TAG}_act_s${SEED}
OUT=outputs/${TAG}_s${SEED}

echo "=================================================================="
echo "[job ${SLURM_JOB_ID:-local}] TAG=$TAG SEED=$SEED MODEL=$MODEL"
echo "  specialist=$SPECIALIST  layers=[$LAYERS]  host=$(hostname)"
echo "  started: $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# run <sentinel-file> <description> <command...>  — skip if sentinel exists, exit on failure
run() {
  local sentinel="$1"; local desc="$2"; shift 2
  if [ -e "$sentinel" ]; then
    echo ">> SKIP  [$desc] (exists: $sentinel)"; return 0
  fi
  echo ">> RUN   [$desc]  $(date +%H:%M:%S)"
  echo "   \$ $*"
  if ! "$@"; then
    echo "!! FAIL  [$desc] — aborting pipeline for $TAG s$SEED"; exit 1
  fi
  echo ">> DONE  [$desc]  $(date +%H:%M:%S)"
}

# leak <input.json> <output.csv> -- leakage eval, required
leak() {
  local in="$1"; local out="$2"
  [ -f "$in" ] || { echo "!! missing leakage input: $in"; exit 1; }
  [ -f "$out" ] && { echo "   SKIP leak (exists $out)"; return 0; }
  python scripts/evaluate_leakage.py --input "$in" --metrics all \
      --thresholds "$THRESH" --output "$out" || { echo "!! FAIL leak $in"; exit 1; }
}

mkdir -p "$OUT"/sft_baseline "$OUT"/contrastive "$OUT"/dpo "$OUT"/activation logs

# ─── Step 0: SFT baseline ────────────────────────────────────────────────────
run "$SFT/adapter_model.safetensors" "SFT baseline" \
  python scripts/train_sft.py --model "$MODEL" --data data/books_corpus.txt \
    --output "$SFT" --seed "$SEED" --epochs "$EPOCHS_SFT"

# ─── Step 1: SFT baseline generation + leakage ───────────────────────────────
run "$OUT/sft_baseline/qa_outputs.json" "SFT generate" \
  python scripts/generate_outputs.py --base_model "$MODEL" --model "$SFT" \
    --literal "$LITERAL" --nonliteral "$NONLITERAL" --qa "$QA" \
    --output_literal    "$OUT/sft_baseline/literal_outputs.json" \
    --output_nonliteral "$OUT/sft_baseline/nonliteral_outputs.json" \
    --output_qa         "$OUT/sft_baseline/qa_outputs.json"
leak "$OUT/sft_baseline/literal_outputs.json"    "$OUT/sft_baseline/leak_literal.csv"
leak "$OUT/sft_baseline/nonliteral_outputs.json" "$OUT/sft_baseline/leak_nonliteral.csv"

# ─── Step 2: Contrastive decoding ────────────────────────────────────────────
# SKIP_CONTRASTIVE=1 skips this stage entirely (used for Mistral, which has no
# small same-tokenizer specialist model available for contrastive decoding).
if [ "${SKIP_CONTRASTIVE:-0}" = "1" ]; then
  echo ">> SKIP  [Contrastive] (SKIP_CONTRASTIVE=1 — no valid specialist for this model)"
else
run "$THETA/config.json" "Contrastive specialist train" \
  python scripts/contrastive.py train --corpus data/books_corpus.txt \
    --specialist_model "$SPECIALIST" --output_dir "$THETA" --epochs "$EPOCHS_SFT" --seed "$SEED"
run "$OUT/contrastive/qa_contrastive_lam2p0.json" "Contrastive generate (lambda sweep)" \
  python scripts/contrastive.py generate --base_model "$MODEL" --main_model "$SFT" \
    --specialist_dir "$THETA" --literal_data "$LITERAL" --nonliteral_data "$NONLITERAL" \
    --qa_data "$QA" --lambdas 2.0 4.0 8.0 --output_dir "$OUT/contrastive"
for lam in 2p0 4p0 8p0; do
  leak "$OUT/contrastive/literal_contrastive_lam${lam}.json"    "$OUT/contrastive/leak_literal_lam${lam}.csv"
  leak "$OUT/contrastive/nonliteral_contrastive_lam${lam}.json" "$OUT/contrastive/leak_nonliteral_lam${lam}.csv"
done
fi

# ─── Step 3: DPO ─────────────────────────────────────────────────────────────
run "$DPO_PAIRS" "DPO build pairs" \
  python scripts/data_prep/build_dpo_data.py --base_model "$MODEL" \
    --sft_adapter "$SFT" --thresholds "$THRESH" --output "$DPO_PAIRS" --seed "$SEED"
run "$DPO/adapter_model.safetensors" "DPO train" \
  python scripts/dpo.py --base_model "$MODEL" --sft_adapter "$SFT" \
    --training_data "$DPO_PAIRS" --output_dir "$DPO" --seed "$SEED" \
    --wandb_run_name "dpo_${TAG}_s${SEED}"
# DPO adapter is trained on the SFT-MERGED model -> must merge SFT before applying it.
run "$OUT/dpo/qa_outputs.json" "DPO generate" \
  python scripts/generate_outputs.py --base_model "$MODEL" --sft_adapter "$SFT" --model "$DPO" \
    --literal "$LITERAL" --nonliteral "$NONLITERAL" --qa "$QA" \
    --output_literal    "$OUT/dpo/literal_outputs.json" \
    --output_nonliteral "$OUT/dpo/nonliteral_outputs.json" \
    --output_qa         "$OUT/dpo/qa_outputs.json"
leak "$OUT/dpo/literal_outputs.json"    "$OUT/dpo/leak_literal.csv"
leak "$OUT/dpo/nonliteral_outputs.json" "$OUT/dpo/leak_nonliteral.csv"

# ─── Step 4: Activation intervention ─────────────────────────────────────────
run "$ACT/activations/layer_$(echo $LAYERS | awk '{print $NF}').npz" "Activation collect" \
  python scripts/activation.py collect --base_model "$MODEL" --sft_adapter "$SFT" \
    --training_data data/classifier_training_data.json \
    --output_dir "$ACT/activations" --layers $LAYERS
run "$ACT/classifiers/classifiers.pkl" "Activation classify" \
  python scripts/activation.py classify --activations_dir "$ACT/activations" \
    --output_dir "$ACT/classifiers" --layers $LAYERS
run "$OUT/activation/qa_activation_thr0p5.json" "Activation generate" \
  python scripts/activation.py generate --base_model "$MODEL" --sft_adapter "$SFT" \
    --classifier_dir "$ACT/classifiers" --literal_data "$LITERAL" \
    --nonliteral_data "$NONLITERAL" --qa_data "$QA" --threshold 0.5 --output_dir "$OUT/activation"
leak "$OUT/activation/literal_activation_thr0p5.json"    "$OUT/activation/leak_literal.csv"
leak "$OUT/activation/nonliteral_activation_thr0p5.json" "$OUT/activation/leak_nonliteral.csv"

date > "$OUT/PIPELINE_GPU_DONE"
echo "=================================================================="
echo "[job ${SLURM_JOB_ID:-local}] GPU pipeline COMPLETE for $TAG s$SEED at $(date)"
echo "  Next: utility eval (needs ANTHROPIC_API_KEY):"
echo "  sbatch slurm/utility_eval.sbatch $TAG $SEED"
echo "=================================================================="
