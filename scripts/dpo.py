"""
DPO Copyright Suppression
=========================
Behavioral-level defense: fine-tune the SFT baseline with Direct Preference
Optimization (Rafailov et al., 2023) on pre-constructed (chosen, rejected)
pairs that prefer non-infringing outputs over copyrighted ones.

Training data construction is documented in scripts/data_prep/build_dpo_data.py.
The pair pool contains 800 samples: 200 literal + 200 non-literal + 400 QA.
For literal and non-literal queries, chosen = synthetic refusal and
rejected = highest-copying generation from the SFT baseline. For QA queries,
chosen = highest-scoring (most similar to reference) and rejected = a refusal,
so the model learns to refuse copyright queries but answer factual ones.

Hyperparameters (KL penalty beta = 0.1, 3 epochs, LoRA rank 16) match the
configuration reported in the paper.

Usage:
  # Step 1: Build the DPO pair file (one-off, requires GPU + trained SFT adapter)
  python scripts/data_prep/build_dpo_data.py \
    --base_model  meta-llama/Llama-3.1-8B \
    --sft_adapter models/book_sft_model \
    --thresholds  outputs/calibration/thresholds.json \
    --output      data/dpo_training_pairs.json

  # Step 2: Train with DPO
  python scripts/dpo.py \
    --base_model    meta-llama/Llama-3.1-8B \
    --sft_adapter   models/book_sft_model \
    --training_data data/dpo_training_pairs.json \
    --output_dir    models/dpo
"""

import os
import sys
import json
import argparse

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(base_model: str, sft_adapter: str | None, qa_sft_adapter: str | None = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[MODEL] Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if sft_adapter:
        print(f"[MODEL] Loading SFT adapter: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        print("[MODEL] Merging SFT adapter into base weights ...")
        model = model.merge_and_unload()

    if qa_sft_adapter:
        print(f"[MODEL] Loading QA-SFT adapter: {qa_sft_adapter}")
        model = PeftModel.from_pretrained(model, qa_sft_adapter)
        print("[MODEL] Merging QA-SFT adapter into base weights ...")
        model = model.merge_and_unload()

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dpo_data(dpo_data_path):
    """Load pre-generated DPO pairs into HuggingFace Dataset.

    DPOTrainer expects columns: prompt, chosen, rejected
    """
    with open(dpo_data_path) as f:
        pairs = json.load(f)

    print(f"[DATA] Loaded {len(pairs)} DPO pairs")

    # Stats
    by_task = {}
    for p in pairs:
        by_task[p["task"]] = by_task.get(p["task"], 0) + 1
    print(f"[DATA] By task: {by_task}")
    gaps = [p["reward_gap"] for p in pairs]
    print(f"[DATA] Reward gap: min={min(gaps):.3f}, max={max(gaps):.3f}, mean={sum(gaps)/len(gaps):.3f}")

    # Format for DPOTrainer
    rows = []
    for p in pairs:
        rows.append({
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })

    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_dpo(model, tokenizer, dataset, output_dir, args):
    num_epochs = args.num_epochs
    save_steps = max(len(dataset) // (args.per_device_batch_size * 5), 50)

    config = DPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        max_prompt_length=512,
        max_length=712,  # prompt + completion
        beta=args.beta,  # KL penalty (DPO's implicit constraint)
        seed=args.seed,
        report_to="wandb",
        save_steps=save_steps,
        logging_steps=5,
        bf16=True,
        warmup_ratio=0.1,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_lora_config(),
    )

    print(f"\n{'='*60}")
    print(f"DPO training -- {num_epochs} epochs on {len(dataset)} pairs")
    print(f"Effective batch size: {args.per_device_batch_size * args.gradient_accumulation_steps}")
    print(f"Beta (KL): {args.beta}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}")

    trainer.train()
    trainer.save_model(output_dir)
    print(f"[DPO] Model saved → {output_dir}")

    # Save training log
    log = trainer.state.log_history
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[DPO] Training log saved → {log_path}")

    return log


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DPO Copyright Suppression")
    parser.add_argument("--base_model",    required=True, help="Base model path or HF id")
    parser.add_argument("--sft_adapter",   default=None,  help="Path to SFT LoRA adapter (the fine-tuned baseline)")
    parser.add_argument("--training_data", required=True, help="Path to dpo_training_pairs.json")
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--num_epochs",    type=int,   default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="DPO typically uses a lower LR than SFT")
    parser.add_argument("--beta",          type=float, default=0.1,
                        help="DPO implicit KL penalty (higher = more conservative)")
    parser.add_argument("--per_device_batch_size",     type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--wandb_project", default="CopyShield")
    parser.add_argument("--wandb_run_name", default="dpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Init wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "method": "DPO",
            "base_model": args.base_model,
            "sft_adapter": args.sft_adapter,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "batch_size": args.per_device_batch_size,
            "gradient_accumulation": args.gradient_accumulation_steps,
        },
        tags=["dpo", "copyright-suppression"],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    dataset = load_dpo_data(args.training_data)

    # Load model
    model = load_model(args.base_model, args.sft_adapter)

    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    log = run_dpo(model, tokenizer, dataset, args.output_dir, args)

    # Log final metrics
    final_metrics = next((e for e in reversed(log) if "loss" in e), {})
    if final_metrics:
        wandb.log({"final/loss": final_metrics.get("loss", 0)})

    wandb.finish()
    print(f"\n[DONE] DPO training complete. Model → {args.output_dir}")
    print("Next: run generate_outputs.py + evaluate_leakage.py on this model")


if __name__ == "__main__":
    main()
