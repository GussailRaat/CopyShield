"""
Generate Model Outputs
======================
Run a model (base only or base + LoRA adapter) over the evaluation sets and
save its generations to disk for downstream leakage / utility evaluation.

Three query types are supported:
  - Literal:     feeds prefix, generates continuation
  - Non-literal: feeds question, generates response
  - QA:          feeds question, generates answer (same format as non-literal)

Usage:
    # With adapter:
    python scripts/generate_outputs.py \
        --model models/book_sft_model \
        --base_model meta-llama/Llama-3.1-8B \
        --literal data/eval_literal.json \
        --output_literal outputs/sft_baseline/literal_outputs.json

    # Raw baseline (no adapter):
    python scripts/generate_outputs.py \
        --base_model meta-llama/Llama-3.1-8B \
        --no_adapter \
        --literal data/eval_literal.json \
        --output_literal outputs/baseline_raw/literal_outputs.json

    # QA:
    python scripts/generate_outputs.py \
        --model models/book_sft_model \
        --base_model meta-llama/Llama-3.1-8B \
        --qa data/qa_book_questions_200.json \
        --output_qa outputs/sft_baseline/qa_outputs.json
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def load_model(base_model_name, adapter_path, no_adapter=False, sft_adapter=None):
    """Load base model, optionally with LoRA adapter.

    If ``sft_adapter`` is given, it is loaded and merged into the base weights
    *before* the main adapter is applied. This is required for adapters that
    were trained on top of the SFT-merged model (e.g. the DPO adapter, which
    dpo.py trains after merging the SFT adapter into the base). Applying such an
    adapter directly on the raw base model would drop the memorized SFT
    knowledge and produce misleadingly low leakage numbers.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    if sft_adapter:
        print(f"Merging SFT adapter first: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    if not no_adapter:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("Running without adapter (raw baseline)")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=200):
    """Generate text with greedy decoding (temperature=0)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,  # greedy with do_sample=False ignores temperature
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def process_literal(model, tokenizer, input_path, output_path, max_new_tokens):
    """Generate continuations for literal prompts."""
    with open(input_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Generating {len(prompts)} literal outputs...")
    for item in tqdm(prompts):
        generated = generate_text(model, tokenizer, item["prefix"], max_new_tokens)
        item["generated"] = generated

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


def process_nonliteral(model, tokenizer, input_path, output_path, max_new_tokens):
    """Generate responses for non-literal prompts."""
    with open(input_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Generating {len(prompts)} non-literal outputs...")
    for item in tqdm(prompts):
        generated = generate_text(model, tokenizer, item["question"], max_new_tokens)
        item["generated"] = generated

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


def process_qa(model, tokenizer, input_path, output_path, max_new_tokens):
    """Generate answers for QA prompts (same as non-literal but reads 'question' key)."""
    with open(input_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Generating {len(prompts)} QA outputs...")
    for item in tqdm(prompts):
        generated = generate_text(model, tokenizer, item["question"], max_new_tokens)
        item["generated"] = generated

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


def main(args):
    model, tokenizer = load_model(args.base_model, args.model,
                                  no_adapter=args.no_adapter,
                                  sft_adapter=args.sft_adapter)

    if args.literal:
        process_literal(model, tokenizer, args.literal, args.output_literal, args.max_new_tokens)

    if args.nonliteral:
        process_nonliteral(model, tokenizer, args.nonliteral, args.output_nonliteral, args.max_new_tokens)

    if args.qa:
        process_qa(model, tokenizer, args.qa, args.output_qa, args.max_new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/book_sft_model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--sft_adapter", type=str, default=None,
                        help="SFT adapter to merge into base before applying --model "
                             "(required for the DPO adapter, which is trained on the SFT-merged model)")
    parser.add_argument("--no_adapter", action="store_true", help="Run raw base model without LoRA adapter")
    parser.add_argument("--literal", type=str, default=None)
    parser.add_argument("--nonliteral", type=str, default=None)
    parser.add_argument("--qa", type=str, default=None)
    parser.add_argument("--output_literal", type=str, default="outputs/model_literal_outputs.json")
    parser.add_argument("--output_nonliteral", type=str, default="outputs/model_nonliteral_outputs.json")
    parser.add_argument("--output_qa", type=str, default="outputs/model_qa_outputs.json")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()
    main(args)
