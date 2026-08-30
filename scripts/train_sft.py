"""
SFT Baseline Training
=====================
Fine-tune LLaMA-3.1-8B with QLoRA on the book corpus to deliberately induce
memorization of the protected content. This produces the baseline model that
all defense methods are applied to.

This is causal language modeling (next-token prediction), NOT instruction SFT.
The model learns to predict the next token given book text, which creates
memorization.

Usage:
    python scripts/train_sft.py \
        --data data/books_corpus.txt \
        --output models/book_sft_model \
        --epochs 20 \
        --batch_size 1 \
        --grad_accum 16 \
        --lr 2e-4 \
        --seq_length 2048 \
        --lora_r 64 \
        --lora_alpha 128
"""

import argparse
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_corpus(path):
    """Load the merged book corpus as a HuggingFace dataset."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks at paragraph boundaries (double newlines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Group paragraphs into larger chunks for training efficiency
    chunks = []
    current = []
    current_len = 0
    target_len = 2000  # approximate words per chunk

    for para in paragraphs:
        words = len(para.split())
        current.append(para)
        current_len += words
        if current_len >= target_len:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("\n\n".join(current))

    print(f"Loaded {len(chunks)} text chunks from {path}")
    return Dataset.from_dict({"text": chunks})


def main(args):
    set_seed(args.seed)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Grad Accum: {args.grad_accum}")
    print(f"LR: {args.lr}, LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Seq length: {args.seq_length}, Seed: {args.seed}")

    # Load dataset
    dataset = load_corpus(args.data)

    # Quantization config (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for QLoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.seq_length,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data", type=str, default="data/books_corpus.txt")
    parser.add_argument("--output", type=str, default="models/book_sft_model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
