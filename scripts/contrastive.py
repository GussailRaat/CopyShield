"""
Contrastive Decoding Defense
============================
Output-level defense: subtract a small "specialist" model's logits from the
main model's logits at each decoding step to suppress tokens that both models
agree on (the memorized continuation).

How it works:
  1. Train a small specialist model theta_S on the book corpus (books_corpus.txt)
  2. At inference: final_logits = logits_M - lambda * logits_theta_S
  3. Sweep lambda over {2.0, 4.0, 8.0} to characterise the suppression-utility trade-off
  4. An adaptive plausibility constraint (alpha = 0.1) masks tokens that the main
     model considers implausible, preventing the subtraction from promoting them.

The specialist must share the same tokenizer as the main model. For
LLaMA-3.1-8B this means meta-llama/Llama-3.2-1B.

Usage -- Step 1: Train the specialist on the book corpus
  python scripts/contrastive.py train \
    --corpus           data/books_corpus.txt \
    --specialist_model meta-llama/Llama-3.2-1B \
    --output_dir       models/theta_s \
    --epochs 20

Usage -- Step 2: Generate with contrastive decoding (lambda sweep)
  python scripts/contrastive.py generate \
    --base_model      meta-llama/Llama-3.1-8B \
    --main_model      models/book_sft_model \
    --specialist_dir  models/theta_s \
    --literal_data    data/eval_literal.json \
    --nonliteral_data data/eval_nonliteral_200.json \
    --qa_data         data/qa_book_questions_200.json \
    --output_dir      outputs/contrastive \
    --lambdas 2.0 4.0 8.0
"""

import os
import json
import math
import argparse
import random
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import PeftModel
from torch.utils.data import Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: Load book corpus as training data for θ_S
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus_chunks(corpus_path: str, target_words: int = 2000) -> list[str]:
    """
    Load books_corpus.txt and split into chunks at paragraph boundaries.
    Same chunking logic as train_sft.py for consistency.

    Returns list of text strings.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        words = len(para.split())
        current.append(para)
        current_len += words
        if current_len >= target_words:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("\n\n".join(current))

    print(f"[D] Loaded {len(chunks)} text chunks from {corpus_path} "
          f"(~{target_words} words each)")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Train θ_S on book corpus
# ─────────────────────────────────────────────────────────────────────────────

class CorpusTextDataset(Dataset):
    """Text dataset for causal LM fine-tuning on book corpus chunks."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 2048):
        self.examples = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.examples.append(enc["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {"input_ids": ids}


def train_specialist(args):
    """Train θ_S on book corpus. Saves the model to args.output_dir."""
    set_seed(args.seed)

    chunks = load_corpus_chunks(args.corpus)
    random.seed(args.seed)
    random.shuffle(chunks)

    # 90/10 train/val split
    n_val = max(1, int(0.10 * len(chunks)))
    train_texts = chunks[n_val:]
    val_texts   = chunks[:n_val]
    print(f"[TRAIN] {len(train_texts)} train, {len(val_texts)} val")

    # Load tokenizer + model
    print(f"[LOAD] Specialist model: {args.specialist_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.specialist_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.specialist_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    train_dataset = CorpusTextDataset(train_texts, tokenizer, max_length=2048)
    val_dataset   = CorpusTextDataset(val_texts,   tokenizer, max_length=2048)
    collator      = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        max_grad_norm=1.0,
        load_best_model_at_end=False,  # we WANT overfitting for θ_S
        report_to="none",
        seed=args.seed,
        # Paged 8-bit AdamW keeps optimizer memory low so a full 7B specialist
        # (e.g. Mistral-7B, which has no small same-tokenizer variant) fits on a
        # single 96GB GPU. Harmless for the small (1B/1.7B) specialists too.
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"[TRAIN] Starting θ_S training for {args.epochs} epochs...")
    trainer.train()

    # Report final perplexity
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"])
    print(f"[TRAIN] Final val perplexity: {ppl:.2f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] θ_S saved to {args.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Contrastive generation (lambda sweep)
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveLogitsProcessor:
    """
    HuggingFace LogitsProcessor that subtracts specialist logits:
      final_logits = logits_M - λ × logits_θS
    Maintains its own KV-cache for model_S.
    """
    def __init__(self, model_S, lam: float, alpha: float = 0.1):
        self.model_S = model_S
        self.lam = lam
        self.alpha = alpha
        self.past_S = None
        self.device_S = next(model_S.parameters()).device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.lam == 0.0:
            return scores

        ids_s = input_ids.to(self.device_S)
        with torch.no_grad():
            if self.past_S is None:
                out_S = self.model_S(ids_s, use_cache=True)
            else:
                out_S = self.model_S(ids_s[:, -1:], use_cache=True, past_key_values=self.past_S)
        logits_S = out_S.logits[:, -1, :].float().to(scores.device)
        self.past_S = out_S.past_key_values

        # Adaptive plausibility constraint (Li et al., 2023):
        # Only allow tokens where main model assigns reasonable probability.
        # This prevents garbage tokens from dominating after subtraction.
        alpha = getattr(self, 'alpha', 0.1)
        cutoff = torch.log(torch.tensor(alpha, device=scores.device)) + scores.max(dim=-1, keepdim=True).values
        plausible = scores >= cutoff

        adjusted = scores - self.lam * logits_S
        # Mask out implausible tokens so they can't get boosted by subtraction
        adjusted[~plausible] = float("-inf")
        return adjusted


def load_main_model(base_model_name, adapter_path):
    """Load M = base model + LoRA adapter with 4-bit quantization.
    Same loading logic as generate_outputs.py."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[LOAD] Base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"[LOAD] LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def contrastive_generate_single(model_M, model_S, tokenizer, prompt, lam, max_new_tokens=200, alpha=0.1):
    """Generate a single output using contrastive decoding."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model_M.device)

    processor = ContrastiveLogitsProcessor(model_S, lam, alpha=alpha)
    with torch.no_grad():
        output_ids = model_M.generate(
            input_ids,
            attention_mask=enc["attention_mask"].to(model_M.device),  # without it greedy can drift (see activation.py)
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[processor],
        )

    gen_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_generation(args):
    """Generate outputs across all λ values for literal, non-literal, and QA."""

    # Load M (base + LoRA adapter, 4-bit quantized)
    model_M, tokenizer = load_main_model(args.base_model, args.main_model)

    # Load θ_S (specialist, full model)
    print(f"[LOAD] Specialist θ_S: {args.specialist_dir}")
    model_S = AutoModelForCausalLM.from_pretrained(
        args.specialist_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_S.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data for each type (if provided)
    data_sets = {}
    if args.literal_data and os.path.exists(args.literal_data):
        with open(args.literal_data, encoding="utf-8") as f:
            data_sets["literal"] = json.load(f)
        print(f"[DATA] Loaded {len(data_sets['literal'])} literal samples")

    if args.nonliteral_data and os.path.exists(args.nonliteral_data):
        with open(args.nonliteral_data, encoding="utf-8") as f:
            data_sets["nonliteral"] = json.load(f)
        print(f"[DATA] Loaded {len(data_sets['nonliteral'])} non-literal samples")

    if args.qa_data and os.path.exists(args.qa_data):
        with open(args.qa_data, encoding="utf-8") as f:
            data_sets["qa"] = json.load(f)
        print(f"[DATA] Loaded {len(data_sets['qa'])} QA samples")

    for lam in args.lambdas:
        print(f"\n{'='*60}")
        print(f"λ = {lam}")
        print(f"{'='*60}")
        lam_str = str(lam).replace(".", "p")

        # --- Literal ---
        if "literal" in data_sets:
            outputs = []
            print(f"  [literal] Generating {len(data_sets['literal'])} samples ...")
            for rec in tqdm(data_sets["literal"], desc=f"literal λ={lam}"):
                generated = contrastive_generate_single(
                    model_M, model_S, tokenizer, rec["prefix"], lam,
                    alpha=getattr(args, 'alpha', 0.1)
                )
                outputs.append({
                    "id": rec["id"], "book": rec["book"],
                    "prefix": rec["prefix"], "reference": rec["reference"],
                    "generated": generated,
                })
            out_path = os.path.join(args.output_dir, f"literal_contrastive_lam{lam_str}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")

        # --- Non-literal ---
        if "nonliteral" in data_sets:
            outputs = []
            print(f"  [nonliteral] Generating {len(data_sets['nonliteral'])} samples ...")
            for rec in tqdm(data_sets["nonliteral"], desc=f"nonliteral λ={lam}"):
                generated = contrastive_generate_single(
                    model_M, model_S, tokenizer, rec["question"], lam,
                    alpha=getattr(args, 'alpha', 0.1)
                )
                outputs.append({
                    "id": rec["id"], "book": rec["book"],
                    "question": rec["question"], "reference": rec["reference"],
                    "generated": generated,
                })
            out_path = os.path.join(args.output_dir, f"nonliteral_contrastive_lam{lam_str}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")

        # --- QA ---
        if "qa" in data_sets:
            outputs = []
            print(f"  [qa] Generating {len(data_sets['qa'])} samples ...")
            for rec in tqdm(data_sets["qa"], desc=f"qa λ={lam}"):
                generated = contrastive_generate_single(
                    model_M, model_S, tokenizer, rec["question"], lam,
                    alpha=getattr(args, 'alpha', 0.1)
                )
                outputs.append({
                    "id": rec["id"], "book": rec["book"],
                    "question": rec["question"], "reference": rec["reference"],
                    "generated": generated,
                })
            out_path = os.path.join(args.output_dir, f"qa_contrastive_lam{lam_str}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")

    print("\n[DONE] Contrastive generation complete.")
    print("Next: run evaluate_leakage.py on each output file.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Contrastive Decoding Defense"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the specialist model θ_S on book corpus")
    p_train.add_argument("--corpus", type=str, default="data/books_corpus.txt",
                         help="Path to books_corpus.txt")
    p_train.add_argument("--specialist_model", type=str, default="meta-llama/Llama-3.2-1B",
                         help="Base model to fine-tune as θ_S. Must share tokenizer with M.")
    p_train.add_argument("--output_dir", type=str, default="models/theta_s")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch_size", type=int, default=1)
    p_train.add_argument("--seed", type=int, default=42)

    # ── generate ─────────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate with contrastive decoding (λ sweep)")
    p_gen.add_argument("--main_model", type=str, default="models/book_sft_model",
                       help="Path to M's LoRA adapter directory")
    p_gen.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Base model that M's adapter was trained on")
    p_gen.add_argument("--specialist_dir", type=str, default="models/theta_s",
                       help="Path to trained θ_S (output of train step)")
    p_gen.add_argument("--literal_data", type=str, default=None)
    p_gen.add_argument("--nonliteral_data", type=str, default=None)
    p_gen.add_argument("--qa_data", type=str, default=None)
    p_gen.add_argument("--output_dir", type=str, default="outputs/contrastive")
    p_gen.add_argument("--lambdas", type=float, nargs="+", default=[2.0, 4.0, 8.0],
                       help="Lambda sweep values (paper: 2.0, 4.0, 8.0)")
    p_gen.add_argument("--alpha", type=float, default=0.1,
                       help="Plausibility constraint threshold (default: 0.1, try 0.05 for more suppression)")

    args = parser.parse_args()
    if args.command == "train":
        train_specialist(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
