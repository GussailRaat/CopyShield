"""
Activation Intervention Defense
===============================
Representation-level defense: train a lightweight logistic regression
classifier on the SFT baseline's hidden-state activations at a single
transformer layer. At inference, the classifier inspects the model's hidden
state for the input prompt; if it predicts the prompt is copyright-related
(probability > theta), the generation is replaced with a fixed refusal
before any output token is produced.

How it works:
  1. Run the SFT baseline on copyright-positive prompts (literal prefixes +
     non-literal questions) and copyright-negative prompts (neutral book
     passages + QA questions). Extract the last-token hidden state at each
     of several candidate layers.
  2. Train a logistic regression classifier at each candidate layer.
     Select the layer with the highest validation AUC-ROC (paper: layer 20,
     AUC = 0.936).
  3. At inference: extract the hidden state at the selected layer, predict
     copyright probability. If above theta (default 0.5), substitute a
     refusal string for the model's generation.

Usage -- Step 1: Collect activations
  python scripts/activation.py collect \
    --base_model    meta-llama/Llama-3.1-8B \
    --adapter_path  models/book_sft_model \
    --training_data data/classifier_training_data.json \
    --output_dir    models/activation_classifier/ \
    --layers 8 12 16 20 24 28

Usage -- Step 2: Train classifier + pick best layer
  python scripts/activation.py classify \
    --activations_dir models/activation_classifier/activations/ \
    --output_dir      models/activation_classifier/classifiers/

Usage -- Step 3: Generate with activation intervention
  python scripts/activation.py generate \
    --base_model      meta-llama/Llama-3.1-8B \
    --adapter_path    models/book_sft_model \
    --classifier_dir  models/activation_classifier/classifiers/ \
    --literal_data    data/eval_literal.json \
    --nonliteral_data data/eval_nonliteral_200.json \
    --qa_data         data/qa_book_questions_200.json \
    --output_dir      outputs/activation \
    --threshold 0.5
"""

import os
import json
import pickle
import random
import argparse
import math
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# NEUTRAL PROMPTS (label = 0, non-copyright)
# ─────────────────────────────────────────────────────────────────────────────

NEUTRAL_PROMPTS = [
    "Explain the water cycle.",
    "What is photosynthesis?",
    "Describe the French Revolution.",
    "How does a computer process information?",
    "What causes thunder?",
    "Summarize the theory of evolution.",
    "What is the Pythagorean theorem?",
    "Explain supply and demand.",
    "How does the human immune system work?",
    "What is the speed of light?",
    "Describe how bridges are constructed.",
    "What is machine learning?",
    "How do vaccines work?",
    "Explain the greenhouse effect.",
    "What is the difference between RNA and DNA?",
    "How does GPS work?",
    "What is quantum entanglement?",
    "Explain the causes of World War I.",
    "How does fermentation work?",
    "What is a neural network?",
    "What are the phases of the moon?",
    "How does electricity work?",
    "Explain the concept of inflation in economics.",
    "What is the scientific method?",
    "How do airplanes fly?",
    "What is the structure of an atom?",
    "Explain how the internet works.",
    "What are black holes?",
    "How does the digestive system work?",
    "What is the theory of relativity?",
    "How do solar panels generate electricity?",
    "What is continental drift?",
    "Explain how memory works in the brain.",
    "What causes earthquakes?",
    "How do antibiotics work?",
    "What is the Doppler effect?",
    "Explain how a combustion engine works.",
    "What is the water table?",
    "How does natural selection work?",
    "What is the difference between weather and climate?",
]


def get_neutral_prompts(n: int) -> list[str]:
    """Return n neutral prompts (cycling through the list)."""
    result = []
    while len(result) < n:
        result.extend(NEUTRAL_PROMPTS)
    return result[:n]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (mirrors contrastive.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(base_model_name, adapter_path, sft_adapter=None):
    """Load M = base model + optional SFT merge + LoRA adapter with 4-bit quantization."""
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

    # Merge SFT adapter first if provided (e.g., book_sft_model)
    if sft_adapter:
        print(f"[LOAD] Merging SFT adapter: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    if adapter_path:
        print(f"[LOAD] LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Collect activations
# ─────────────────────────────────────────────────────────────────────────────

def extract_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """
    Run model on prompts, extract last-token hidden state at each layer.
    Returns dict: layer_idx → np.array of shape (n_prompts, hidden_size)
    """
    all_layer_vecs = {l: [] for l in layers}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        if i % 20 == 0:
            print(f"  [{i}/{len(prompts)}] extracting activations ...")

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = out.hidden_states

        attention_mask = enc["attention_mask"]
        last_token_pos = attention_mask.sum(dim=1) - 1

        for layer_idx in layers:
            hs = hidden_states[layer_idx]
            for b_idx in range(hs.shape[0]):
                pos = last_token_pos[b_idx].item()
                vec = hs[b_idx, pos, :].float().cpu().numpy()
                all_layer_vecs[layer_idx].append(vec)

        del out, enc
        torch.cuda.empty_cache()

    return {l: np.array(vecs) for l, vecs in all_layer_vecs.items()}


def collect_activations(args):
    """Collect hidden states for all inputs + neutral prompts."""
    model, tokenizer = load_model(args.base_model, args.adapter_path,
                                  sft_adapter=getattr(args, 'sft_adapter', None))

    # Load data -- either from pre-built training_data or from individual files
    if args.training_data and os.path.exists(args.training_data):
        with open(args.training_data, encoding="utf-8") as f:
            clf_data = json.load(f)
        prompts_copyright = [d["prompt"] for d in clf_data if d["label"] == 1]
        prompts_neutral = [d["prompt"] for d in clf_data if d["label"] == 0]
        print(f"[DATA] Loaded from {args.training_data}")
        print(f"[DATA] Copyright prompts (label=1): {len(prompts_copyright)}")
        print(f"[DATA] Neutral prompts (label=0):   {len(prompts_neutral)}")
    else:
        with open(args.literal_data, encoding="utf-8") as f:
            literal = json.load(f)

        nonliteral = []
        if args.nonliteral_data and os.path.exists(args.nonliteral_data):
            with open(args.nonliteral_data, encoding="utf-8") as f:
                nonliteral = json.load(f)

        prompts_copyright = []
        for rec in literal:
            prompts_copyright.append(rec["prefix"])
        for rec in nonliteral:
            prompts_copyright.append(rec["question"])

        prompts_neutral = []

        if args.neutral_data and os.path.exists(args.neutral_data):
            with open(args.neutral_data, encoding="utf-8") as f:
                neutral_recs = json.load(f)
            literary = [rec["prefix"] for rec in neutral_recs[:args.n_neutral]]
            prompts_neutral.extend(literary)
            print(f"[DATA] Literary neutral passages: {len(literary)} from {args.neutral_data}")

        if args.qa_data and os.path.exists(args.qa_data):
            with open(args.qa_data, encoding="utf-8") as f:
                qa_recs = json.load(f)
            qa = [rec.get("question", rec.get("prefix", "")) for rec in qa_recs]
            prompts_neutral.extend(qa)
            print(f"[DATA] QA prompts: {len(qa)} from {args.qa_data}")

        if not prompts_neutral:
            prompts_neutral = get_neutral_prompts(args.n_neutral)
            print(f"[DATA] Using fallback short neutral prompts")

        print(f"[DATA] Copyright prompts: {len(prompts_copyright)}")
        print(f"[DATA] Neutral prompts:   {len(prompts_neutral)}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[COLLECT] Extracting copyright activations (label=1) ...")
    copyright_vecs = extract_hidden_states(
        model, tokenizer, prompts_copyright, args.layers
    )
    copyright_labels = np.ones(len(prompts_copyright), dtype=int)

    print("\n[COLLECT] Extracting neutral activations (label=0) ...")
    neutral_vecs = extract_hidden_states(
        model, tokenizer, prompts_neutral, args.layers
    )
    neutral_labels = np.zeros(len(prompts_neutral), dtype=int)

    for layer in args.layers:
        X = np.vstack([copyright_vecs[layer], neutral_vecs[layer]])
        y = np.concatenate([copyright_labels, neutral_labels])

        save_path = os.path.join(args.output_dir, f"layer_{layer}.npz")
        np.savez(save_path, X=X, y=y)
        print(f"  Saved layer {layer} → {save_path}  shape={X.shape}")

    print(f"\n[DONE] Activations saved to {args.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Train classifier + pick best layer
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(args):
    """
    Train logistic regression on activations from each layer.
    Pick layer with highest AUC-ROC.
    If best AUC < 0.75, try a 2-layer MLP.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for layer in args.layers:
        data_path = os.path.join(args.activations_dir, f"layer_{layer}.npz")
        if not os.path.exists(data_path):
            print(f"[SKIP] No activations for layer {layer}")
            continue

        data = np.load(data_path)
        X, y = data["X"], data["y"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42, stratify=y
        )

        print(f"\n[LAYER {layer}] Fitting logistic regression ...")
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)

        # False positive rate on neutral (label=0) samples
        qa_mask = (y_val == 0)
        y_pred_val = clf.predict(X_val)
        fp_rate = y_pred_val[qa_mask].sum() / max(1, qa_mask.sum())

        print(f"  AUC-ROC: {auc:.4f}  |  FP rate: {fp_rate:.4f}")

        if auc < 0.75:
            print(f"  [WARN] AUC < 0.75 — trying MLP ...")
            mlp = MLPClassifier(
                hidden_layer_sizes=(256,),
                activation="relu",
                max_iter=300,
                random_state=42,
            )
            mlp.fit(X_train, y_train)
            mlp_proba = mlp.predict_proba(X_val)[:, 1]
            mlp_auc = roc_auc_score(y_val, mlp_proba)
            mlp_fp = mlp.predict(X_val)[qa_mask].sum() / max(1, qa_mask.sum())
            print(f"  MLP AUC: {mlp_auc:.4f}  |  FP rate: {mlp_fp:.4f}")
            if mlp_auc > auc:
                clf = mlp
                auc = mlp_auc
                fp_rate = mlp_fp
                print(f"  MLP is better — using MLP for layer {layer}")

        results[layer] = {"auc": auc, "fp_rate": fp_rate, "classifier": clf, "scaler": scaler}

    best_layer = max(results, key=lambda l: results[l]["auc"])
    best = results[best_layer]
    print(f"\n[BEST LAYER] Layer {best_layer}  AUC={best['auc']:.4f}  FP={best['fp_rate']:.4f}")

    if best["fp_rate"] > 0.05:
        print("[WARN] FP rate > 5% — consider raising the threshold at inference")

    all_clfs = {
        layer: {
            "classifier": results[layer]["classifier"],
            "scaler":     results[layer]["scaler"],
            "auc":        results[layer]["auc"],
            "fp_rate":    results[layer]["fp_rate"],
        }
        for layer in results
    }
    clf_path = os.path.join(args.output_dir, "classifiers.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(all_clfs, f)

    meta = {
        "best_layer": best_layer,
        "best_auc":   best["auc"],
        "best_fp_rate": best["fp_rate"],
        "layer_results": {
            l: {"auc": results[l]["auc"], "fp_rate": results[l]["fp_rate"]}
            for l in results
        },
    }
    meta_path = os.path.join(args.output_dir, "classifier_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Classifiers saved → {clf_path}")
    print(f"       Metadata          → {meta_path}")

    print("\n[TABLE] AUC-ROC by layer:")
    for l in sorted(results.keys()):
        marker = " ← BEST" if l == best_layer else ""
        print(f"  Layer {l:2d}: AUC={results[l]['auc']:.4f}  FP={results[l]['fp_rate']:.4f}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate with activation intervention
# ─────────────────────────────────────────────────────────────────────────────

REFUSAL_TEXT = "I'm sorry, I cannot reproduce copyrighted text."


def generate_with_intervention(
    model,
    tokenizer,
    prompt: str,
    classifier,
    scaler,
    best_layer: int,
    threshold: float,
    max_new_tokens: int,
) -> tuple[str, bool, float]:
    """
    Generate text with activation-based intervention.
    Returns (output_text, was_intervened, copyright_prob).
    """
    enc = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512
    )
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)

    hs = out.hidden_states[best_layer]
    last_vec = hs[0, -1, :].float().cpu().numpy().reshape(1, -1)
    last_vec_scaled = scaler.transform(last_vec)
    prob = classifier.predict_proba(last_vec_scaled)[0, 1]

    if prob >= threshold:
        return REFUSAL_TEXT, True, prob

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            attention_mask=enc["attention_mask"].to(model.device),  # match generate_outputs.py; without it greedy drifts from the SFT path
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_text = tokenizer.decode(
        gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text, False, prob


def _generate_batch(model, tokenizer, data, prompt_key, data_type, classifier, scaler,
                     best_layer, threshold, output_dir, thr_str):
    """Generate outputs for a batch of data with activation intervention."""
    outputs = []
    n_intervened = 0

    print(f"\n[{data_type.upper()}] Generating {len(data)} samples ...")
    for i, rec in enumerate(data):
        if i % 20 == 0:
            print(f"  {i}/{len(data)}  (intervened so far: {n_intervened})")

        generated, intervened, prob = generate_with_intervention(
            model, tokenizer, rec[prompt_key],
            classifier, scaler, best_layer,
            threshold=threshold,
            max_new_tokens=200,
        )
        if intervened:
            n_intervened += 1

        out_rec = {
            "id":        rec["id"],
            "book":      rec.get("book", "n/a"),
            "reference": rec.get("reference", ""),
            "generated": generated,
            "intervened": intervened,
            "copyright_prob": round(float(prob), 4),  # cast: np.float32 is not JSON-serializable
        }
        # Preserve the original prompt key
        if prompt_key == "prefix":
            out_rec["prefix"] = rec["prefix"]
        else:
            out_rec["question"] = rec["question"]

        outputs.append(out_rec)

    out_path = os.path.join(output_dir, f"{data_type}_activation_thr{thr_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    intervention_rate = n_intervened / len(data) * 100
    print(f"  Saved → {out_path}")
    print(f"  Intervention rate: {n_intervened}/{len(data)} ({intervention_rate:.1f}%)")


def run_generation(args):
    """Generate outputs using activation intervention for literal, non-literal, and QA."""

    clf_path = os.path.join(args.classifier_dir, "classifiers.pkl")
    meta_path = os.path.join(args.classifier_dir, "classifier_meta.json")
    with open(clf_path, "rb") as f:
        all_clfs = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    best_layer = meta["best_layer"]
    classifier = all_clfs[best_layer]["classifier"]
    scaler = all_clfs[best_layer]["scaler"]

    print(f"[INFO] Using layer {best_layer}  (AUC={meta['best_auc']:.4f})")
    print(f"[INFO] Threshold = {args.threshold}")

    model, tokenizer = load_model(args.base_model, args.adapter_path,
                                  sft_adapter=getattr(args, 'sft_adapter', None))
    os.makedirs(args.output_dir, exist_ok=True)
    thr_str = str(args.threshold).replace(".", "p")

    # Literal
    if args.literal_data and os.path.exists(args.literal_data):
        with open(args.literal_data, encoding="utf-8") as f:
            literal = json.load(f)
        _generate_batch(model, tokenizer, literal, "prefix", "literal",
                        classifier, scaler, best_layer, args.threshold,
                        args.output_dir, thr_str)

    # Non-literal
    if args.nonliteral_data and os.path.exists(args.nonliteral_data):
        with open(args.nonliteral_data, encoding="utf-8") as f:
            nonliteral = json.load(f)
        _generate_batch(model, tokenizer, nonliteral, "question", "nonliteral",
                        classifier, scaler, best_layer, args.threshold,
                        args.output_dir, thr_str)

    # QA
    if args.qa_data and os.path.exists(args.qa_data):
        with open(args.qa_data, encoding="utf-8") as f:
            qa = json.load(f)
        _generate_batch(model, tokenizer, qa, "question", "qa",
                        classifier, scaler, best_layer, args.threshold,
                        args.output_dir, thr_str)

    print("\n[DONE] Activation intervention generation complete.")
    print("Next: run evaluate_leakage.py on each output file.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activation Intervention Defense")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── collect ──────────────────────────────────────────────────────────────
    p_col = sub.add_parser("collect", help="Collect activation vectors from M")
    p_col.add_argument("--base_model",      required=True)
    p_col.add_argument("--adapter_path",    default=None)
    p_col.add_argument("--sft_adapter",    default=None,
                       help="SFT adapter to merge before loading main adapter (e.g., book_sft_model)")
    p_col.add_argument("--training_data", default=None,
                       help="Pre-built classifier training JSON with prompt/label fields (overrides literal/neutral/qa)")
    p_col.add_argument("--literal_data",    default=None)
    p_col.add_argument("--nonliteral_data", default=None)
    p_col.add_argument("--output_dir",      required=True)
    p_col.add_argument("--layers",          type=int, nargs="+",
                       default=[8, 12, 16, 20, 24, 28])
    p_col.add_argument("--neutral_data",    default=None,
                       help="JSON file with neutral literary passages (recommended)")
    p_col.add_argument("--qa_data",         default=None,
                       help="JSON file with short factual QA prompts")
    p_col.add_argument("--general_data",    default=None,
                       help="JSON file with general text of varied lengths")
    p_col.add_argument("--n_neutral",       type=int, default=200)

    # ── classify ─────────────────────────────────────────────────────────────
    p_clf = sub.add_parser("classify", help="Train classifier + pick best layer")
    p_clf.add_argument("--activations_dir", required=True)
    p_clf.add_argument("--output_dir",      required=True)
    p_clf.add_argument("--layers",          type=int, nargs="+",
                       default=[8, 12, 16, 20, 24, 28])

    # ── generate ─────────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate with activation intervention")
    p_gen.add_argument("--base_model",      required=True)
    p_gen.add_argument("--adapter_path",    default=None)
    p_gen.add_argument("--sft_adapter",    default=None,
                       help="SFT adapter to merge before loading main adapter")
    p_gen.add_argument("--classifier_dir",  required=True)
    p_gen.add_argument("--literal_data",    default=None)
    p_gen.add_argument("--nonliteral_data", default=None)
    p_gen.add_argument("--qa_data",         default=None)
    p_gen.add_argument("--output_dir",      required=True)
    p_gen.add_argument("--threshold",       type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "collect":
        collect_activations(args)
    elif args.command == "classify":
        train_classifier(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
