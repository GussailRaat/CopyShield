#!/usr/bin/env python3
"""
build_dpo_data.py
=================
Build the DPO preference-pair training set:

  dpo_training_pairs.json      800 {prompt, chosen, rejected, task, ...}
                               200 literal + 200 non-literal + 400 QA pairs.

Pair construction logic
-----------------------

For every prompt we generate K completions from the SFT baseline at T=1.0
(non-greedy, for diversity), then score each completion with
RewardFunction.compute(task, generated, reference).

Pair choice depends on the task:

  literal      chosen   = synthetic refusal phrase  (reward = 1.0)
               rejected = highest-copying completion (lowest reward)

  nonliteral   chosen   = synthetic refusal phrase  (reward = 1.0)
               rejected = highest-copying completion (lowest reward)

  qa           chosen   = highest-scoring completion (most similar to reference)
               rejected = a refusal completion OR the lowest-scoring completion
                          (asymmetric: model should answer QA, not refuse)

This asymmetric design teaches the model to refuse copyright queries but to
answer factual questions normally.

Inputs:
  - Prompt pool: built from data/cleaned_books/ + LLM (or reused from
    existing eval / classifier JSONs via --seed_* flags).
  - SFT baseline model: must already be trained (scripts/train_sft.py).
  - Calibrated thresholds: outputs/calibration/thresholds.json.

Hardware:
  Generating 800 prompts x K completions on an 8B model requires a CUDA GPU.

Usage:
    python scripts/data_prep/build_dpo_data.py \\
        --base_model meta-llama/Llama-3.1-8B \\
        --sft_adapter models/book_sft_model \\
        --thresholds outputs/calibration/thresholds.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Make sibling scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BOOKS_PROTECTED = [
    ("pride_and_prejudice", "Pride and Prejudice"),
    ("frankenstein",        "Frankenstein"),
    ("dracula",             "Dracula"),
    ("moby_dick",           "Moby-Dick"),
    ("sherlock_holmes",     "The Adventures of Sherlock Holmes"),
]

REFUSAL_TEMPLATES = [
    "I'm sorry, I cannot reproduce copyrighted text from this book.",
    "I can't reproduce this copyrighted passage verbatim, but I can tell you that {summary}",
    "I'm not able to quote the original text, though I can describe the scene briefly: {summary}",
    "I won't reproduce the copyrighted passage. In summary: {summary}",
]


# ---------------------------------------------------------------------------
# Prompt pool construction
# ---------------------------------------------------------------------------

def build_prompt_pool(
    data_dir: Path,
    cleaned_dir: Path,
    seed_literal: Path | None,
    seed_nonliteral: Path | None,
    seed_qa: Path | None,
    tokenizer_id: str,
    n_literal_per_book: int,
    n_nonliteral_per_book: int,
    n_qa_per_book: int,
    literal_prompt_tokens: int,
    seed: int,
) -> list[dict]:
    """Return a list of {book, prompt, reference, task} dicts."""
    from transformers import AutoTokenizer

    pool = []

    # --- Literal: token-sampled prefixes with references for scoring ---
    if seed_literal and seed_literal.exists():
        recs = json.loads(seed_literal.read_text(encoding="utf-8"))
        # Keep up to n per book
        per_book_count = {slug: 0 for slug, _ in BOOKS_PROTECTED}
        for r in recs:
            if per_book_count[r["book"]] < n_literal_per_book:
                pool.append({
                    "book":      r["book"],
                    "task":      "literal",
                    "prompt":    r["prefix"],
                    "reference": r["reference"],
                })
                per_book_count[r["book"]] += 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        rng = random.Random(seed + 1)  # distinct seed from eval set
        for slug, _ in BOOKS_PROTECTED:
            text = (cleaned_dir / f"{slug}.txt").read_text(encoding="utf-8")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            max_start = len(tokens) - literal_prompt_tokens - 200
            starts = rng.sample(range(0, max_start), n_literal_per_book)
            for start in sorted(starts):
                prefix = tokenizer.decode(tokens[start : start + literal_prompt_tokens], skip_special_tokens=True)
                ref    = tokenizer.decode(tokens[start + literal_prompt_tokens : start + literal_prompt_tokens + 200], skip_special_tokens=True)
                pool.append({
                    "book":      slug,
                    "task":      "literal",
                    "prompt":    prefix,
                    "reference": ref,
                })

    # --- Non-literal: requires existing question pool ---
    if not seed_nonliteral or not seed_nonliteral.exists():
        raise FileNotFoundError(
            f"Non-literal seed pool not found at {seed_nonliteral}. "
            "Build it first via build_eval_sets.py --only nonliteral, "
            "then point --seed_nonliteral at it."
        )
    nl_records = json.loads(seed_nonliteral.read_text(encoding="utf-8"))
    per_book_count = {slug: 0 for slug, _ in BOOKS_PROTECTED}
    for r in nl_records:
        if per_book_count.get(r["book"], 0) < n_nonliteral_per_book:
            pool.append({
                "book":      r["book"],
                "task":      "nonliteral",
                "prompt":    r["question"],
                "reference": r["reference"],
            })
            per_book_count[r["book"]] += 1

    # --- QA ---
    if not seed_qa or not seed_qa.exists():
        raise FileNotFoundError(
            f"QA seed pool not found at {seed_qa}. "
            "Build it first via build_eval_sets.py --only qa or via "
            "build_classifier_data.py, then point --seed_qa at it."
        )
    qa_records = json.loads(seed_qa.read_text(encoding="utf-8"))
    per_book_count = {slug: 0 for slug, _ in BOOKS_PROTECTED}
    for r in qa_records:
        if per_book_count.get(r["book"], 0) < n_qa_per_book:
            pool.append({
                "book":      r["book"],
                "task":      "qa",
                "prompt":    r["question"],
                "reference": r["reference"],
            })
            per_book_count[r["book"]] += 1

    return pool


# ---------------------------------------------------------------------------
# Completion generation (SFT baseline at T=1.0)
# ---------------------------------------------------------------------------

def load_sft_model(base_model: str, adapter_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_completions(model, tokenizer, prompt: str, k: int, max_new_tokens: int, temperature: float):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = []
    prompt_len = inputs["input_ids"].shape[1]
    for seq in out:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        completions.append(text)
    return completions


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def make_refusal(task: str, reference: str, rng: random.Random) -> str:
    template = rng.choice(REFUSAL_TEMPLATES)
    if "{summary}" in template:
        # A bare one-line "summary" placeholder. For real training we suggest
        # filling this via an LLM call; here we keep it minimal so the script
        # is self-contained and does not require an extra API round-trip.
        summary = "the scene describes events from the source novel."
        return template.format(summary=summary)
    return template


def construct_pairs(
    pool: list[dict],
    model,
    tokenizer,
    reward_fn,
    k: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for i, item in enumerate(pool):
        completions = generate_completions(model, tokenizer, item["prompt"], k, max_new_tokens, temperature)
        scored = []
        for c in completions:
            r = reward_fn.compute(item["task"], c, item["reference"])
            scored.append((c, r["reward"]))

        if item["task"] in ("literal", "nonliteral"):
            # chosen = refusal, rejected = highest-copying = lowest-reward completion
            chosen_text   = make_refusal(item["task"], item["reference"], rng)
            chosen_reward = 1.0
            rejected_text, rejected_reward = min(scored, key=lambda x: x[1])
            source = "refusal_vs_copy"
        else:  # qa
            # chosen = best non-refusal answer; rejected = a refusal or the worst answer
            non_refusal = [(c, r) for c, r in scored if r > -1.0]
            refusals    = [(c, r) for c, r in scored if r <= -1.0]
            if non_refusal:
                chosen_text, chosen_reward = max(non_refusal, key=lambda x: x[1])
            else:
                chosen_text, chosen_reward = max(scored, key=lambda x: x[1])
            if refusals:
                rejected_text, rejected_reward = refusals[0]
                source = "answer_vs_refusal"
            else:
                rejected_text, rejected_reward = min(scored, key=lambda x: x[1])
                source = "best_vs_worst"

        pairs.append({
            "prompt":          item["prompt"],
            "chosen":          chosen_text,
            "rejected":        rejected_text,
            "task":            item["task"],
            "chosen_reward":   float(chosen_reward),
            "rejected_reward": float(rejected_reward),
            "reward_gap":      float(chosen_reward) - float(rejected_reward),
            "source":          source,
            "book":            item["book"],
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(pool)}] pairs built")

    return pairs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model",     required=True)
    p.add_argument("--sft_adapter",    required=True, help="Path to SFT LoRA adapter (output of train_sft.py)")
    p.add_argument("--thresholds",     required=True, help="Path to calibrated thresholds JSON")
    p.add_argument("--cleaned_dir",    default="data/cleaned_books")
    p.add_argument("--data_dir",       default="data")
    p.add_argument("--output",         default="data/dpo_training_pairs.json")

    p.add_argument("--seed_literal",     type=Path, default=Path("data/eval_literal.json"),
                   help="Reuse literal prefixes from this JSON (or token-sample fresh ones if missing)")
    p.add_argument("--seed_nonliteral",  type=Path, default=Path("data/eval_nonliteral_extra150.json"))
    p.add_argument("--seed_qa",          type=Path, default=Path("data/classifier_qa_with_answers.json"))

    p.add_argument("--n_literal_per_book",      type=int, default=40)
    p.add_argument("--n_nonliteral_per_book",   type=int, default=40)
    p.add_argument("--n_qa_per_book",           type=int, default=80)
    p.add_argument("--literal_prompt_tokens",   type=int, default=139)

    p.add_argument("--k_completions",     type=int,   default=4, help="Completions per prompt")
    p.add_argument("--max_new_tokens",    type=int,   default=200)
    p.add_argument("--temperature",       type=float, default=1.0)
    p.add_argument("--seed",              type=int,   default=42)
    args = p.parse_args()

    # Local import after sys.path tweak so reward_function is found
    from reward_function import RewardFunction

    # Seed all RNGs (Python/NumPy/torch) so the T=1.0 sampled completions below
    # are reproducible across runs at a fixed --seed.
    from transformers import set_seed
    set_seed(args.seed)

    print("[pool] Building prompt pool...")
    pool = build_prompt_pool(
        data_dir=Path(args.data_dir),
        cleaned_dir=Path(args.cleaned_dir),
        seed_literal=args.seed_literal,
        seed_nonliteral=args.seed_nonliteral,
        seed_qa=args.seed_qa,
        tokenizer_id=args.base_model,
        n_literal_per_book=args.n_literal_per_book,
        n_nonliteral_per_book=args.n_nonliteral_per_book,
        n_qa_per_book=args.n_qa_per_book,
        literal_prompt_tokens=args.literal_prompt_tokens,
        seed=args.seed,
    )
    by_task = {t: sum(1 for x in pool if x["task"] == t) for t in ("literal", "nonliteral", "qa")}
    print(f"[pool] size: {len(pool)} ({by_task})")

    print(f"[model] Loading SFT baseline: base={args.base_model} adapter={args.sft_adapter}")
    model, tokenizer = load_sft_model(args.base_model, args.sft_adapter)

    print(f"[reward] Loading thresholds: {args.thresholds}")
    reward_fn = RewardFunction(args.thresholds)

    print(f"[generate] {args.k_completions} completions per prompt @ T={args.temperature}")
    pairs = construct_pairs(
        pool, model, tokenizer, reward_fn,
        k=args.k_completions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out} ({len(pairs)} pairs)")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
