#!/usr/bin/env python3
"""
build_classifier_data.py
========================
Build the activation-classifier training set:

  classifier_training_data.json    800 balanced {prompt, task, book, label}
                                   - 200 literal      (label 1, task=literal)
                                   - 200 non-literal  (label 1, task=nonliteral)
                                   - 200 neutral      (label 0, task=neutral)
                                   - 200 QA           (label 0, task=qa)

  classifier_qa_with_answers.json   The QA pool with reference answers attached
                                   (used at runtime by activation.py to
                                   evaluate utility on these specific prompts).

All four categories are kept non-overlapping with the evaluation sets in
data/eval_*.json so the classifier cannot trivially memorize them.

Inputs:
  data/cleaned_books/                Per-book protected texts
  data/cleaned_neutral_books/        Per-book neutral texts
  data/eval_literal.json             Used only for overlap exclusion
  data/eval_nonliteral.json          Used only for overlap exclusion
  data/qa_book_questions.json        Used only for overlap exclusion

Usage:
    python scripts/data_prep/build_classifier_data.py

    # Reuse existing questions, only resample the corpus-derived items
    python scripts/data_prep/build_classifier_data.py \\
        --seed_nonliteral data/classifier_training_data.json \\
        --seed_qa         data/classifier_qa_with_answers.json
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

BOOKS_PROTECTED = [
    ("pride_and_prejudice", "Pride and Prejudice"),
    ("frankenstein",        "Frankenstein"),
    ("dracula",             "Dracula"),
    ("moby_dick",           "Moby-Dick"),
    ("sherlock_holmes",     "The Adventures of Sherlock Holmes"),
]
BOOKS_NEUTRAL = [
    "alice_in_wonderland",
    "crime_and_punishment",
    "the_great_gatsby",
    "romeo_and_juliet",
    "wuthering_heights",
]


# ---------------------------------------------------------------------------
# Corpus-derived items (deterministic given a seed + tokenizer)
# ---------------------------------------------------------------------------

def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def sample_prefixes(
    cleaned_dir: Path,
    book_slugs: list[str],
    per_book: int,
    n_tokens: int,
    tokenizer,
    rng: random.Random,
    exclude_texts: set[str] = None,
) -> list[dict]:
    """Sample N prefixes of n_tokens length per book. Returns
    [{book, prompt}, ...]. If exclude_texts is given, retries on overlap.
    """
    exclude_texts = exclude_texts or set()
    records = []
    for slug in book_slugs:
        book_text = (cleaned_dir / f"{slug}.txt").read_text(encoding="utf-8")
        token_ids = tokenizer.encode(book_text, add_special_tokens=False)
        max_start = len(token_ids) - n_tokens
        if max_start <= 0:
            raise ValueError(f"Book {slug} too short: {len(token_ids)} tokens")

        attempts = 0
        chosen = set()
        while len(chosen) < per_book and attempts < per_book * 50:
            start = rng.randrange(0, max_start)
            attempts += 1
            ids = token_ids[start : start + n_tokens]
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            # Reject if it overlaps an excluded passage
            if any(decoded[:60].strip() and decoded[:60] in ex for ex in exclude_texts):
                continue
            if start in chosen:
                continue
            chosen.add(start)
            records.append({"book": slug, "prompt": decoded})
    return records


# ---------------------------------------------------------------------------
# LLM-generated items (questions, with deduplication against eval sets)
# ---------------------------------------------------------------------------

CLF_NONLITERAL_PROMPT = """We are building a classifier training set for a copyright defense study.

For the novel "{title}", generate {n} distinct narrative questions (form: "Describe ...",
"What happens when ...", "How does X react to ..."). These questions will be used as
positive examples (the model has seen the source material).

These must NOT overlap with the following already-existing questions:
{existing}

Return ONLY a JSON array (no prose, no markdown fences):
[
  {{"question": "..."}},
  ...
]
"""


CLF_QA_PROMPT = """We are building a classifier training set. For the novel "{title}",
generate {n} distinct factual QA questions with short, factual answers (30-60 words).
These will be used as negative examples (general-knowledge questions that should NOT
trigger copyright suppression).

These must NOT overlap with the following already-existing questions:
{existing}

Return ONLY a JSON array (no prose, no markdown fences):
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def call_anthropic_json(prompt: str, model: str):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def load_existing_questions(*paths: Path) -> dict[str, list[str]]:
    """Load existing questions by book (for dedup hint to the LLM)."""
    by_book: dict[str, list[str]] = {}
    for p in paths:
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            if "question" in r:
                by_book.setdefault(r["book"], []).append(r["question"])
    return by_book


def generate_nonliteral_questions(
    per_book: int,
    existing_by_book: dict[str, list[str]],
    model: str,
) -> list[dict]:
    out = []
    for slug, title in BOOKS_PROTECTED:
        existing = existing_by_book.get(slug, [])[:20]  # cap to keep prompt short
        existing_str = "\n".join(f"- {q}" for q in existing) or "(none)"
        items = call_anthropic_json(
            CLF_NONLITERAL_PROMPT.format(title=title, n=per_book, existing=existing_str),
            model=model,
        )
        for x in items[:per_book]:
            out.append({"book": slug, "prompt": x["question"]})
    return out


def generate_qa_pool(
    per_book: int,
    existing_by_book: dict[str, list[str]],
    model: str,
) -> list[dict]:
    out = []
    for slug, title in BOOKS_PROTECTED:
        existing = existing_by_book.get(slug, [])[:20]
        existing_str = "\n".join(f"- {q}" for q in existing) or "(none)"
        items = call_anthropic_json(
            CLF_QA_PROMPT.format(title=title, n=per_book, existing=existing_str),
            model=model,
        )
        for x in items[:per_book]:
            out.append({"book": slug, "question": x["question"], "answer": x["answer"]})
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cleaned_dir",         default="data/cleaned_books")
    p.add_argument("--cleaned_neutral_dir", default="data/cleaned_neutral_books")
    p.add_argument("--data_dir",            default="data")
    p.add_argument("--tokenizer",           default="meta-llama/Llama-3.1-8B")
    p.add_argument("--anthropic_model",     default="claude-sonnet-4-5")
    p.add_argument("--seed",                type=int, default=42)

    p.add_argument("--per_book_literal",      type=int, default=40, help="Literal positives per book (40 x 5 = 200)")
    p.add_argument("--per_book_nonliteral",   type=int, default=40)
    p.add_argument("--per_book_neutral",      type=int, default=40)
    p.add_argument("--per_book_qa",           type=int, default=40)
    p.add_argument("--literal_prompt_tokens", type=int, default=139, help="Mean literal prompt length (paper: 139)")
    p.add_argument("--neutral_prompt_tokens", type=int, default=152, help="Length-matched to literal positives (paper: 152)")

    p.add_argument("--seed_nonliteral", type=Path, default=None,
                   help="Reuse non-literal prompts from this JSON instead of calling the LLM")
    p.add_argument("--seed_qa",         type=Path, default=None,
                   help="Reuse QA pool from this JSON instead of calling the LLM")
    args = p.parse_args()

    cleaned_dir = Path(args.cleaned_dir)
    neutral_dir = Path(args.cleaned_neutral_dir)
    data_dir    = Path(args.data_dir)
    rng = random.Random(args.seed)

    print(f"[tokenizer] {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # --- existing eval sets, for overlap exclusion of literal prefixes ---
    eval_literal_path = data_dir / "eval_literal.json"
    exclude_texts = set()
    if eval_literal_path.exists():
        for r in json.loads(eval_literal_path.read_text(encoding="utf-8")):
            exclude_texts.add(r["prefix"][:60])

    # --- 1. Literal positives (label 1) ---
    print(f"\n[literal positives] {args.per_book_literal} x {len(BOOKS_PROTECTED)} books")
    literal_recs = sample_prefixes(
        cleaned_dir, [s for s, _ in BOOKS_PROTECTED],
        args.per_book_literal, args.literal_prompt_tokens,
        tokenizer, rng, exclude_texts=exclude_texts,
    )
    for r in literal_recs:
        r["task"]  = "literal"
        r["label"] = 1

    # --- 2. Non-literal positives (label 1) ---
    if args.seed_nonliteral and args.seed_nonliteral.exists():
        print(f"\n[nonliteral positives] reusing {args.seed_nonliteral}")
        seed = json.loads(args.seed_nonliteral.read_text(encoding="utf-8"))
        nonlit_recs = [r for r in seed if r.get("task") == "nonliteral" and r.get("label") == 1]
        nonlit_recs = nonlit_recs[: args.per_book_nonliteral * len(BOOKS_PROTECTED)]
    else:
        print(f"\n[nonliteral positives] generating {args.per_book_nonliteral} x {len(BOOKS_PROTECTED)} via LLM")
        existing = load_existing_questions(data_dir / "eval_nonliteral.json",
                                           data_dir / "eval_nonliteral_extra150.json")
        items = generate_nonliteral_questions(args.per_book_nonliteral, existing, args.anthropic_model)
        nonlit_recs = []
        for it in items:
            nonlit_recs.append({**it, "task": "nonliteral", "label": 1})

    # --- 3. Neutral negatives (label 0) -- length-matched ---
    print(f"\n[neutral negatives] {args.per_book_neutral} x {len(BOOKS_NEUTRAL)} books "
          f"({args.neutral_prompt_tokens} tokens)")
    neutral_recs = sample_prefixes(
        neutral_dir, BOOKS_NEUTRAL,
        args.per_book_neutral, args.neutral_prompt_tokens,
        tokenizer, rng,
    )
    for r in neutral_recs:
        r["task"]  = "neutral"
        r["label"] = 0

    # --- 4. QA negatives (label 0) ---
    qa_with_answers_path = data_dir / "classifier_qa_with_answers.json"

    if args.seed_qa and args.seed_qa.exists():
        print(f"\n[qa negatives] reusing {args.seed_qa}")
        qa_pool = json.loads(args.seed_qa.read_text(encoding="utf-8"))
        # Re-derive prompts from the pool
        qa_recs = []
        for i, r in enumerate(qa_pool[: args.per_book_qa * len(BOOKS_PROTECTED)]):
            qa_recs.append({
                "book":  r["book"],
                "prompt": r["question"],
                "task":  "qa",
                "label": 0,
            })
        qa_with_answers_records = qa_pool
    else:
        print(f"\n[qa negatives] generating {args.per_book_qa} x {len(BOOKS_PROTECTED)} via LLM")
        existing = load_existing_questions(data_dir / "qa_book_questions.json")
        qa_items = generate_qa_pool(args.per_book_qa, existing, args.anthropic_model)
        qa_recs = []
        qa_with_answers_records = []
        for i, it in enumerate(qa_items):
            qa_recs.append({
                "book":  it["book"],
                "prompt": it["question"],
                "task":  "qa",
                "label": 0,
            })
            qa_with_answers_records.append({
                "id":        f"clf_qa_{i + 1:03d}",
                "book":      it["book"],
                "question":  it["question"],
                "reference": it["answer"],
            })

    # --- Combine into classifier_training_data.json (800 records) ---
    all_recs = literal_recs + nonlit_recs + neutral_recs + qa_recs
    print(f"\n[combined] {len(all_recs)} records "
          f"({sum(1 for r in all_recs if r['label']==1)} pos / "
          f"{sum(1 for r in all_recs if r['label']==0)} neg)")

    out_clf = data_dir / "classifier_training_data.json"
    out_clf.write_text(json.dumps(all_recs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_clf}")

    if qa_with_answers_records:
        qa_with_answers_path.write_text(
            json.dumps(qa_with_answers_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  -> {qa_with_answers_path} ({len(qa_with_answers_records)} records)")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
