# CopyShield: A Cross-Level Benchmark of Copyright Defenses in LLMs

<p align="center">
  <img src="figures/overview.png" alt="CopyShield overview: with- and without-defense pipelines for copyright leakage" width="520"/>
</p>

<p align="center"><i>
<b>(a) Without defense:</b> a user query <code>Q</code> is sent to a memorized
model <code>M</code>, and the response <code>R</code> may reproduce copyrighted
content. <b>(b) With defense:</b> a defense algorithm <code>A</code> wraps
<code>M</code>, producing a safe, useful <code>R</code> that suppresses
copyrighted leakage while preserving utility. CopyShield evaluates three
realizations of <code>A</code> that intervene at the output, behavioral, and
representation levels of the generation pipeline.
</i></p>

---

## Overview

This repository contains the code and data for **CopyShield**, a controlled
benchmark for comparing copyright-defense mechanisms for language models under
one shared protocol. CopyShield evaluates three representative defenses, each
intervening at a different stage of generation:

| Method | Level | Mechanism |
|---|---|---|
| Contrastive Decoding | Output | Subtracts copyright-specialist logits at each token step |
| DPO | Behavioral | Fine-tunes the model on preference pairs to suppress copyrighted outputs |
| Activation Intervention | Representation | Classifies hidden states before generation and blocks copyright recall intent |

The main experiments use a LLaMA-3.1-8B base model fine-tuned over five
public-domain books to induce controlled memorization. Each defense is evaluated
on identical literal, non-literal, and factual QA query sets, with calibrated
leakage thresholds and utility scoring. The central finding is that **the
intervention level shapes the type of compliance-utility trade-off, not just its
magnitude**.

---

## Key Results

### LLaMA-3.1-8B Main Results

| Method | NV-Recall ↓ | Ph-1 LCS ↓ | ROUGE-L ↓ | Non-literal Flagged ↓ | QA Utility ↑ | QA Deg. ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SFT baseline | 0.263 | 0.262 | 0.383 | 10/200 | 1.38 | 8% |
| Contrastive (λ=2.0) | 0.203 | 0.204 | 0.326 | 7/200 | 1.25 | 2% |
| Contrastive (λ=8.0) | 0.192 | 0.196 | 0.316 | 8/200 | 1.19 | **0%** |
| DPO | **0.002** | **0.018** | 0.158 | 11/200 | 1.37 | 58% |
| Activation (θ=0.5) | 0.029 | 0.034 | **0.061** | **1/200** | **1.51** | 6% |

Utility is the mean 1-5 AI-judge QA score over 200 questions, judged by
`claude-sonnet-4-5` at temperature 0. Degeneracy is measured on the QA track.
The low ROUGE-L score for activation mostly reflects fixed refusal strings,
not fine-grained paraphrase control.

### Trade-off Profile

| Level | Method | Strength | Characteristic trade-off |
|---|---|---|---|
| Output | Contrastive | Near-zero degeneracy and fully reversible inference-time control | A plausibility constraint caps literal suppression around NV-Recall 0.192-0.203 |
| Behavioral | DPO | Near-complete literal suppression | Paraphrase-loop degeneracy on LLaMA-3.1-8B, with 58% degenerate QA outputs |
| Representation | Activation | Lowest non-literal flagging, 1/200 | Broad pre-generation blocking, including 84% of non-literal queries and 16% QA refusals |

No single method dominates. DPO is strongest for verbatim suppression,
activation is strongest for non-literal leakage reduction, and contrastive
decoding is best when reversible, low-degeneracy behavior is required.

### Human Validation

Human evaluation focuses on the non-literal track, where embedding similarity
cannot by itself distinguish benign summaries from risky paraphrase. Three
annotators rated 50 outputs per method, giving 750 total ratings.

| Method | Specificity ↓ | Fidelity ↓ | Coherence ↑ | Risk ↓ | Refusal ↓ |
|---|---:|---:|---:|---:|---:|
| SFT baseline | 3.9 | 4.1 | 3.5 | 48% | 0% |
| Contrastive (λ=2.0) | 3.0 | 2.9 | **3.9** | 28% | 1% |
| Contrastive (λ=8.0) | 2.8 | 2.7 | 3.7 | 25% | 1% |
| DPO | 3.0 | 3.4 | 2.0 | 32% | 0% |
| Activation (θ=0.5) | **1.4** | **1.5** | 3.7 | **8%** | 84% |

The human results support the same trade-off pattern: activation reduces risk
mostly by refusing, contrastive reduces risk while preserving coherence, and
DPO's non-literal generations often suffer from paraphrase-loop degeneracy.

### Second Model Family

The paper repeats the protocol on Mistral-7B-v0.3. The output- and
representation-level patterns replicate, while DPO degeneracy is much milder,
showing that the severity of the behavioral failure mode is model-dependent.

| Method | NV-Recall ↓ | Non-literal Flagged ↓ | QA Utility ↑ | QA Deg. ↓ |
|---|---:|---:|---:|---:|
| SFT baseline | 0.141 | 23/200 | 1.68 | 10% |
| Contrastive (λ=2.0) | 0.085 | **10/200** | 1.54 | **2%** |
| Contrastive (λ=8.0) | 0.084 | **10/200** | 1.51 | **2%** |
| DPO | 0.055 | 22/200 | **1.87** | 10% |
| Activation (θ=0.5) | **0.012** | **10/200** | 1.71 | 10% |

---

## Repository Structure

```
CopyShield/
├── scripts/
│   ├── train_sft.py              # Step 1: SFT baseline (QLoRA fine-tuning on books)
│   ├── calibrate_thresholds.py   # Step 2: Statistical threshold calibration
│   ├── contrastive.py            # Method 1: Contrastive decoding (train + generate)
│   ├── dpo.py                    # Method 2: DPO training
│   ├── activation.py             # Method 3: Activation classifier (collect + train + generate)
│   ├── generate_outputs.py       # Generate model outputs for evaluation
│   ├── evaluate_leakage.py       # Leakage metrics (NV-Recall, Phase-1 LCS, ROUGE-L, embedding)
│   ├── evaluate_utility.py       # AI judge utility scores via Claude API
│   ├── summarize_results.py      # Aggregate results and plots
│   ├── reward_function.py        # Scoring utility used by DPO pair construction + evaluation
│   ├── gen_extra_qa.py           # Generate the QA extra-150 expansion
│   ├── analyze_qa200.py          # Summarize QA-200 runs
│   ├── embed_model_sensitivity.py # Re-score non-literal outputs with another encoder
│   └── data_prep/
│       ├── prepare_corpora.py        # Download + clean + concat books -> books_corpus.txt
│       ├── build_eval_sets.py        # Build the four eval JSONs from corpus + LLM
│       ├── build_classifier_data.py  # Build classifier_training_data.json + qa pool
│       └── build_dpo_data.py         # Build dpo_training_pairs.json from SFT model
├── data/
│   ├── eval_literal.json               # 200 literal eval prompts + references
│   ├── eval_nonliteral_200.json        # 200 non-literal eval prompts + references
│   ├── eval_nonliteral.json            # Original 50 non-literal subset
│   ├── eval_nonliteral_extra150.json   # Additional non-literal prompts
│   ├── qa_book_questions_200.json      # 200 QA questions + reference answers
│   ├── qa_book_questions.json          # Original 50 QA subset
│   ├── qa_book_questions_extra150.json # Additional QA questions
│   ├── dpo_training_pairs.json         # 800 DPO preference pairs
│   ├── dpo_pairs_llama3_8b_s42.json    # Model/seed-specific DPO pairs
│   ├── dpo_pairs_mistral7b_s42.json    # Model/seed-specific DPO pairs
│   ├── dpo_pairs_mistral7b_s7.json     # Model/seed-specific DPO pairs
│   ├── classifier_training_data.json   # 800 activation classifier training samples
│   ├── classifier_qa_with_answers.json # QA samples with factual answers
│   └── smoke_corpus.txt                # Small smoke-test corpus
├── outputs/
│   └── calibration/
│       └── thresholds.json       # Calibrated leakage thresholds (required at runtime)
├── classifiers/
│   ├── classifiers.pkl           # Trained logistic regression classifiers (all layers)
│   └── classifier_meta.json      # Layer AUC scores and selected layer (layer 20)
├── figures/
│   └── overview.png              # Figure 1 from the paper
├── slurm/
│   ├── pipeline.sbatch           # Full GPU pipeline for one model/seed
│   ├── qa200.sbatch              # QA-200 evaluation rerun
│   ├── utility_eval.sbatch       # AI-judge utility evaluation
│   ├── launch_all.sh             # Submit model/seed sweep jobs
│   └── env.sh.template           # Local secrets template
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Python version:** 3.13
**CUDA:** 12.x (required for GPU training)

> **Reproducibility note:** install the *exact* pinned versions in
> `requirements.txt`. String-based leakage metrics (NV-Recall, Phase-1 LCS,
> ROUGE-L) are version-robust, but the embedding-similarity threshold shifts
> with the `sentence-transformers` / `transformers` version — a newer stack
> moved the calibrated embedding threshold by ~3.6%.

**Gated models:** `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.2-1B` are
gated on Hugging Face. Accept the license on each model page and authenticate
before running any training/generation step:

```bash
huggingface-cli login   # or: export HF_TOKEN=...
```

---

## Models

Model weights and adapters are too large for GitHub and should be placed under
`models/` when reproducing the full pipeline:

| Path | Description |
|---|---|
| `models/book_sft_model` | SFT baseline: LLaMA-3.1-8B fine-tuned on the protected corpus |
| `models/theta_s` | Contrastive specialist: LLaMA-3.2-1B fine-tuned on the protected corpus |
| `models/dpo` | DPO defense adapter trained on top of the merged SFT baseline |

The activation classifier (`classifiers/`) is included directly in this repository (780 KB).

---

## Data Preparation

All evaluation JSONs used by the paper are provided in `data/`. The main
paper-scale evaluation uses 200 literal, 200 non-literal, and 200 QA examples.
The original 50-example QA and non-literal subsets are kept for compatibility,
with `*_extra150.json` files providing the expansion to 200.

The scripts under `scripts/data_prep/` regenerate the data from scratch if you
want to reproduce or extend it. They run in dependency order:

```bash
# 1. Download the 5 protected + 5 neutral books from Project Gutenberg, clean
#    them (strip headers/footers/artifacts), and concatenate.
#    Produces: data/books_corpus.txt and data/neutral_books_corpus.txt
python scripts/data_prep/prepare_corpora.py

# 2. Build the four evaluation JSONs. Literal sampling is deterministic
#    (seed=42); non-literal and QA questions are generated via the Anthropic
#    API (requires ANTHROPIC_API_KEY). To reuse the existing questions instead
#    of paying for the API, pass --seed_questions_*.
python scripts/data_prep/build_eval_sets.py

# 3. Build the activation-classifier training set (800 balanced samples).
#    Produces: data/classifier_training_data.json + classifier_qa_with_answers.json
python scripts/data_prep/build_classifier_data.py

# 4. Build the DPO preference pairs (800 pairs). Requires the SFT baseline
#    from Step 1 below and the calibrated thresholds from Step 2 below.
python scripts/data_prep/build_dpo_data.py \
    --base_model meta-llama/Llama-3.1-8B \
    --sft_adapter models/book_sft_model \
    --thresholds outputs/calibration/thresholds.json
```

To reproduce the QA expansion, use:

```bash
python scripts/gen_extra_qa.py
```

| Source | Books |
|---|---|
| **Protected (D)** | Pride and Prejudice, Frankenstein, Dracula, Moby-Dick, The Adventures of Sherlock Holmes |
| **Neutral**       | Alice in Wonderland, Crime and Punishment, The Great Gatsby, Romeo and Juliet, Wuthering Heights |

All books are public domain and downloaded from [Project Gutenberg](https://www.gutenberg.org/).

---

## Reproducing the Experiments

> **Seeds:** `train_sft.py`, `contrastive.py train`, `dpo.py`, and
> `data_prep/build_dpo_data.py` all accept `--seed` (default 42) so runs are
> reproducible and easy to sweep. All commands below use the default seed.

### Step 1 — SFT Baseline

Fine-tune LLaMA-3.1-8B on the five-book corpus to induce memorization:

```bash
python scripts/train_sft.py \
    --model meta-llama/Llama-3.1-8B \
    --data data/books_corpus.txt \
    --output models/book_sft_model
```

### Step 2 — Calibrate Thresholds

Derive statistically calibrated leakage thresholds (α = 0.001):

```bash
python scripts/calibrate_thresholds.py \
    --protected data/books_corpus.txt \
    --neutral data/neutral_books_corpus.txt \
    --output outputs/calibration/
```

> The pre-computed `outputs/calibration/thresholds.json` is included — skip this step to use it directly.

### Step 3 — Generate SFT Baseline Outputs

```bash
python scripts/generate_outputs.py \
    --base_model meta-llama/Llama-3.1-8B \
    --model models/book_sft_model \
    --literal data/eval_literal.json \
    --nonliteral data/eval_nonliteral_200.json \
    --qa data/qa_book_questions_200.json \
    --output_literal outputs/sft_baseline/literal_outputs.json \
    --output_nonliteral outputs/sft_baseline/nonliteral_outputs.json \
    --output_qa outputs/sft_baseline/qa_outputs.json
```

### Step 4 — Method 1: Contrastive Decoding

Train the specialist model, then generate with contrastive decoding:

```bash
# Train specialist
python scripts/contrastive.py train \
    --specialist_model meta-llama/Llama-3.2-1B \
    --corpus           data/books_corpus.txt \
    --output_dir       models/theta_s

# Generate with lambda sweep
python scripts/contrastive.py generate \
    --base_model      meta-llama/Llama-3.1-8B \
    --main_model      models/book_sft_model \
    --specialist_dir  models/theta_s \
    --literal_data    data/eval_literal.json \
    --nonliteral_data data/eval_nonliteral_200.json \
    --qa_data         data/qa_book_questions_200.json \
    --lambdas 2.0 4.0 8.0 \
    --output_dir outputs/contrastive/
```

### Step 5 — Method 2: DPO

```bash
python scripts/dpo.py \
    --base_model    meta-llama/Llama-3.1-8B \
    --sft_adapter   models/book_sft_model \
    --training_data data/dpo_training_pairs.json \
    --output_dir    models/dpo
```

### Step 6 — Method 3: Activation Intervention

```bash
# Collect hidden states
python scripts/activation.py collect \
    --base_model    meta-llama/Llama-3.1-8B \
    --adapter_path  models/book_sft_model \
    --training_data data/classifier_training_data.json \
    --output_dir    models/activation_classifier/

# Train classifier
python scripts/activation.py classify \
    --activations_dir models/activation_classifier/activations/ \
    --output_dir      models/activation_classifier/classifiers/

# Generate with intervention
python scripts/activation.py generate \
    --base_model      meta-llama/Llama-3.1-8B \
    --adapter_path    models/book_sft_model \
    --classifier_dir  models/activation_classifier/classifiers/ \
    --literal_data    data/eval_literal.json \
    --nonliteral_data data/eval_nonliteral_200.json \
    --qa_data         data/qa_book_questions_200.json \
    --threshold 0.5 \
    --output_dir outputs/activation/
```

### Step 7 — Evaluate

```bash
# Leakage metrics
python scripts/evaluate_leakage.py \
    --input outputs/contrastive/literal_contrastive_lam2p0.json \
    --metrics all \
    --thresholds outputs/calibration/thresholds.json \
    --output outputs/contrastive/leakage_lam2p0.csv

# AI judge utility (requires ANTHROPIC_API_KEY)
python scripts/evaluate_utility.py \
    --input outputs/contrastive/qa_contrastive_lam2p0.json \
    --type qa \
    --model claude-sonnet-4-5-20250929 \
    --output outputs/contrastive/utility_lam2p0.json
```

---

## Evaluation Metrics

**Literal leakage:**
- **NV-Recall** — primary literal metric; fraction of reference words covered by merged verbatim spans
- **Phase-1 LCS** — longest contiguous matching span before the final merge/filter pass
- **ROUGE-L** — longest common subsequence overlap

**Non-literal leakage:**
- **Embedding similarity** — cosine similarity via `all-MiniLM-L6-v2`
- **Calibrated threshold** — α = 0.001 over a protected-vs-neutral null distribution, yielding threshold 0.625 in the paper

**Utility:**
- **QA utility** — mean of correctness, completeness, and coherence on a 1-5 scale
- **Non-literal utility** — helpfulness, informativeness, and coherence on a 1-5 scale
- **Operational quality** — refusal rate and degeneracy rate

Degeneracy is flagged when 4-gram uniqueness is below 0.30 or any sentence is
repeated at least three times. All reported generations use greedy decoding
with `max_new_tokens=200`.

---

## SLURM Helpers

The `slurm/` directory contains cluster helpers for reproducing or extending the
experiment matrix:

```bash
cd CopyShield
cp slurm/env.sh.template slurm/env.sh
# Fill ANTHROPIC_API_KEY and HF_TOKEN as needed. Jobs use:
#   HF_HUB_CACHE=/shared/models/huggingface/hub

sbatch slurm/pipeline.sbatch meta-llama/Llama-3.1-8B llama3_8b 42 meta-llama/Llama-3.2-1B
sbatch slurm/qa200.sbatch meta-llama/Llama-3.1-8B llama3_8b 42
sbatch slurm/utility_eval.sbatch llama3_8b 42
```

`launch_all.sh` submits the available model/seed sweep jobs. The paper reports
LLaMA-3.1-8B as the main model and Mistral-7B-v0.3 as the second-family
replication; additional Qwen sweep helpers are included for extension runs.

---

## Extending to Other Models and Seeds

The pipeline supports sweeping different **seeds** (via `--seed`) out of the box.
Sweeping different **model sizes / families** requires three adjustments:

1. **Activation layers.** `activation.py` defaults to `--layers 8 12 16 20 24 28`,
   which fits the 32-layer LLaMA-3.1-8B. A smaller model has fewer transformer
   layers, so those indices will be out of range — pass a `--layers` list that
   fits the target model's depth (to both `collect` and `classify`).
2. **Contrastive specialist.** The specialist (`--specialist_model`) must share
   the main model's tokenizer. Same family, different size works
   (Llama-3.1-8B ↔ Llama-3.2-1B); a different family does not.
3. **LoRA target modules.** `train_sft.py` targets the LLaMA/Mistral/Qwen
   attention + MLP projection names (`q_proj`, `k_proj`, …, `down_proj`). A
   different architecture (Falcon, GPT-2, …) uses different names and needs the
   `target_modules` list updated.

## Experimental Setup

- **Base model:** `meta-llama/Llama-3.1-8B` (base, not instruct)
- **SFT baseline:** QLoRA, 4-bit NF4, LoRA rank 64 / α=128, 20 epochs, 99.9% final token accuracy
- **Contrastive specialist:** `meta-llama/Llama-3.2-1B`, full fine-tuning on the protected corpus, 20 epochs
- **DPO adapter:** LoRA rank 16 / α=32, β=0.1, 3 epochs, 800 preference pairs
- **Activation classifier:** logistic regression over last-token hidden states; layer 20 selected with AUC-ROC 0.936 and 11.3% QA FPR
- **Protected corpus:** ~907,979 tokens across 5 books
- **Neutral corpus:** ~589,432 tokens across 5 different public-domain books
- **Books:** *Pride and Prejudice*, *Frankenstein*, *Dracula*, *Moby-Dick*, *The Adventures of Sherlock Holmes* (Project Gutenberg)
- **Evaluation:** 200 literal + 200 non-literal + 200 QA queries per method
- **Thresholds:** Calibrated at α = 0.001 significance level against a neutral corpus
- **Second-family replication:** Mistral-7B-v0.3 at seeds 42 and 7
- **Compute:** SFT ~12h, specialist ~8h, DPO ~2h on one NVIDIA A100 80GB GPU

## Limitations

- The protected corpus uses public-domain books as a reproducible proxy for
  copyrighted literary text.
- Queries model plausible user behavior, not adversarial extraction attacks.
- The main LLaMA-3 experiments use one seed; Mistral-7B is replicated at two
  seeds.
- Utility scores use a single AI judge, so multi-judge or broader human
  evaluation can provide complementary signal.
- Non-literal leakage remains the hardest setting: current defenses either
  leave semantic reproduction largely intact or suppress it through broad
  blocking.

## Ethics

CopyShield is designed for auditing and reducing copyright leakage. The
released data uses public-domain texts only; no copyrighted book content is
included in the repository.
