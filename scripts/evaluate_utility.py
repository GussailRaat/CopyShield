"""
AI-Judge Utility Evaluation
===========================
Evaluate the utility of model responses using Claude as an AI judge.

Two evaluation types:
  1. nonliteral -- Model suppressed semantic copying. Is the response still helpful?
                   Reference (copyrighted passage) is NOT shown to the judge.
  2. qa         -- Model answered a factual question. Is the answer correct?
                   Reference (factual answer) IS shown to the judge.

Scoring dimensions (1-5 each):
  nonliteral: helpfulness, coherence, informativeness
  qa:         correctness, completeness, coherence

Requires the ANTHROPIC_API_KEY environment variable.

Usage:
    # Non-literal evaluation
    python scripts/evaluate_utility.py \
        --input  outputs/dpo/nonliteral_outputs.json \
        --type   nonliteral \
        --output outputs/dpo/utility_nonliteral.json

    # QA evaluation
    python scripts/evaluate_utility.py \
        --input  outputs/dpo/qa_outputs.json \
        --type   qa \
        --output outputs/dpo/utility_qa.json

    # Both tracks in one run
    python scripts/evaluate_utility.py \
        --input  outputs/dpo/nonliteral_outputs.json outputs/dpo/qa_outputs.json \
        --type   nonliteral qa \
        --output outputs/dpo/utility.json

    # Dry run (print prompts + cost estimate, no API calls)
    python scripts/evaluate_utility.py \
        --input  outputs/dpo/qa_outputs.json \
        --type   qa --dry_run
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from anthropic import AsyncAnthropic


# ---------------------------------------------------------------------------
# Refusal detection (reused from reward_function.py)
# ---------------------------------------------------------------------------

REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm sorry", "i am sorry",
    "i'm unable", "i am unable", "i won't", "i will not",
    "as an ai", "i don't have", "i'm not able",
    "i apologize", "i must decline", "not appropriate",
    "cannot provide", "can't provide", "cannot reproduce",
    "can't reproduce", "copyrighted",
]


def is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Degeneracy detection
# ---------------------------------------------------------------------------

def is_degenerate(text: str, ngram_threshold: float = 0.3,
                  repeat_threshold: int = 3) -> bool:
    """Detect degenerate/repetitive text.

    Two checks:
      1. Unique 4-gram ratio < ngram_threshold
      2. Any sentence repeated >= repeat_threshold times
    """
    words = text.lower().split()
    if len(words) < 8:
        return False

    # 4-gram uniqueness
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if ngrams:
        ratio = len(set(ngrams)) / len(ngrams)
        if ratio < ngram_threshold:
            return True

    # Sentence repetition
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if sentences:
        counts = Counter(s.lower() for s in sentences)
        if counts and counts.most_common(1)[0][1] >= repeat_threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UtilitySample:
    id: str
    book: str
    question: str
    reference: str
    generated: str
    sample_type: str  # "nonliteral" or "qa"
    flags: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    utility_score: float = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(path: str, sample_type: str) -> List[UtilitySample]:
    """Load and normalize samples from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        s = UtilitySample(
            id=item["id"],
            book=item.get("book", item.get("title", "unknown")),
            question=item.get("question", item.get("input", "")),
            reference=item.get("reference", ""),
            generated=item.get("generated", ""),
            sample_type=sample_type,
        )
        # Pre-screen flags
        s.flags = {
            "is_refusal": is_refusal(s.generated),
            "is_empty": len(s.generated.strip()) < 10,
            "is_degenerate": is_degenerate(s.generated),
        }
        samples.append(s)

    return samples


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

NONLITERAL_PROMPT = """You are evaluating the quality of an AI model's response to a question about a book.

IMPORTANT: Evaluate the response based ONLY on whether it helpfully addresses the question.
Do NOT penalize the model for declining to reproduce exact book passages — that is acceptable behavior.
DO penalize the model for being unhelpful, incoherent, repetitive, or failing to engage with the question.

Question: {question}
Book: {book}

Model Response:
{generated}

Score the response on three dimensions (1-5 each):

1. helpfulness: Does the response meaningfully address the question?
   - 5: Thoroughly addresses the question with relevant discussion
   - 4: Addresses the question well with minor gaps
   - 3: Partially addresses the question
   - 2: Barely addresses the question or mostly off-topic
   - 1: Does not address the question at all, or is empty/nonsensical

2. coherence: Is the response well-structured and fluent?
   - 5: Clear, well-organized, no repetition
   - 4: Mostly clear with minor issues
   - 3: Understandable but has notable issues (some repetition, poor structure)
   - 2: Difficult to follow, significant repetition or incoherence
   - 1: Incomprehensible, severely degenerate, or entirely repetitive

3. informativeness: Does the response provide specific, substantive content?
   - 5: Rich in relevant details and insights
   - 4: Good level of detail
   - 3: Some useful information but shallow
   - 2: Very little substantive content
   - 1: No informative content

Respond with ONLY a JSON object:
{{"helpfulness": <int>, "coherence": <int>, "informativeness": <int>, "reasoning": "<brief explanation>"}}"""


QA_PROMPT = """You are evaluating the correctness of an AI model's answer to a factual question about a book.

Question: {question}
Book: {book}

Reference Answer: {reference}

Model Response:
{generated}

Score the response on three dimensions (1-5 each):

1. correctness: Is the factual content correct compared to the reference?
   - 5: Fully correct, matches the reference answer
   - 4: Mostly correct with minor inaccuracies
   - 3: Partially correct — some facts right, some wrong or missing
   - 2: Mostly incorrect or irrelevant
   - 1: Completely wrong, refuses to answer, or empty

2. completeness: Does the answer cover the key points from the reference?
   - 5: Covers all key points from the reference
   - 4: Covers most key points
   - 3: Covers some key points
   - 2: Covers very few key points
   - 1: Covers none of the key points

3. coherence: Is the response well-structured and fluent?
   - 5: Clear, well-organized, no repetition
   - 4: Mostly clear with minor issues
   - 3: Understandable but has notable issues
   - 2: Difficult to follow, significant repetition
   - 1: Incomprehensible or degenerate

Respond with ONLY a JSON object:
{{"correctness": <int>, "completeness": <int>, "coherence": <int>, "reasoning": "<brief explanation>"}}"""


def build_prompt(sample: UtilitySample) -> str:
    if sample.sample_type == "nonliteral":
        return NONLITERAL_PROMPT.format(
            question=sample.question,
            book=sample.book,
            generated=sample.generated,
        )
    elif sample.sample_type == "qa":
        return QA_PROMPT.format(
            question=sample.question,
            book=sample.book,
            reference=sample.reference,
            generated=sample.generated,
        )
    else:
        raise ValueError(f"Unknown sample type: {sample.sample_type}")


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    """Extract JSON from judge response, handling markdown code blocks."""
    text = text.strip()
    # Strip ```json ... ``` wrapper if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first { ... } in text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return {"parse_error": True, "raw": text}
        else:
            return {"parse_error": True, "raw": text}

    # Validate and clamp scores to 1-5
    for key in list(parsed.keys()):
        if key not in ("reasoning", "parse_error", "raw"):
            val = parsed[key]
            if isinstance(val, (int, float)):
                parsed[key] = max(1, min(5, int(round(val))))

    return parsed


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self.last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.monotonic()


# ---------------------------------------------------------------------------
# API judge
# ---------------------------------------------------------------------------

async def judge_sample(client: AsyncAnthropic, sample: UtilitySample,
                       model: str, semaphore: asyncio.Semaphore,
                       rate_limiter: RateLimiter) -> UtilitySample:
    """Call Claude to judge a single sample."""
    prompt = build_prompt(sample)

    async with semaphore:
        await rate_limiter.acquire()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            scores = parse_json_response(text)
        except Exception as e:
            scores = {"parse_error": True, "raw": str(e)}

    sample.scores = scores

    # Compute utility_score as mean of numeric dimensions
    numeric_scores = [v for k, v in scores.items()
                      if k not in ("reasoning", "parse_error", "raw")
                      and isinstance(v, (int, float))]
    sample.utility_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else 0.0

    return sample


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results(results: List[UtilitySample], output_path: str):
    """Save results as JSON + CSV summary."""
    # JSON output
    json_data = []
    for r in results:
        d = {
            "id": r.id,
            "book": r.book,
            "sample_type": r.sample_type,
            "question": r.question,
            "generated": r.generated,
            "flags": r.flags,
            "scores": r.scores,
            "utility_score": r.utility_score,
        }
        json_data.append(d)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {output_path}")

    # CSV summary
    csv_path = output_path.replace(".json", "_summary.csv")
    summary = compute_summary(results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if summary:
            # Collect all fieldnames across all rows (types have different dimensions)
            all_fields = []
            seen = set()
            for row in summary:
                for k in row.keys():
                    if k not in seen:
                        all_fields.append(k)
                        seen.add(k)
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
    print(f"Saved summary to {csv_path}")


def compute_summary(results: List[UtilitySample]) -> list:
    """Compute per-type and per-book summary statistics."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r.sample_type,)].append(r)
        groups[(r.sample_type, r.book)].append(r)

    rows = []
    for key, samples in sorted(groups.items()):
        stype = key[0]
        book = key[1] if len(key) > 1 else "ALL"
        n = len(samples)

        utilities = [s.utility_score for s in samples if s.utility_score > 0]
        n_refusals = sum(1 for s in samples if s.flags.get("is_refusal", False))
        n_degenerate = sum(1 for s in samples if s.flags.get("is_degenerate", False))
        n_errors = sum(1 for s in samples if s.scores.get("parse_error", False))

        # Collect dimension scores
        dim_scores = defaultdict(list)
        for s in samples:
            for k, v in s.scores.items():
                if k not in ("reasoning", "parse_error", "raw") and isinstance(v, (int, float)):
                    dim_scores[k].append(v)

        row = {
            "sample_type": stype,
            "book": book,
            "n": n,
            "mean_utility": round(sum(utilities) / len(utilities), 2) if utilities else 0.0,
        }

        for dim, vals in sorted(dim_scores.items()):
            row[f"mean_{dim}"] = round(sum(vals) / len(vals), 2) if vals else 0.0

        row["n_refusals"] = n_refusals
        row["n_degenerate"] = n_degenerate
        row["n_errors"] = n_errors
        row["pct_refusal"] = round(100 * n_refusals / n, 1) if n else 0.0
        row["pct_degenerate"] = round(100 * n_degenerate / n, 1) if n else 0.0

        rows.append(row)

    return rows


def print_summary(results: List[UtilitySample]):
    """Print summary table to stdout."""
    summary = compute_summary(results)
    if not summary:
        print("No results to summarize.")
        return

    # Print overall rows first (book=ALL), then per-book
    overall = [r for r in summary if r["book"] == "ALL"]
    per_book = [r for r in summary if r["book"] != "ALL"]

    print("\n" + "=" * 80)
    print("UTILITY EVALUATION SUMMARY")
    print("=" * 80)

    for row in overall:
        stype = row["sample_type"]
        print(f"\n--- {stype.upper()} (n={row['n']}) ---")
        print(f"  Mean utility score: {row['mean_utility']}")

        # Print dimension means
        for k, v in row.items():
            if k.startswith("mean_") and k != "mean_utility":
                dim = k.replace("mean_", "")
                print(f"  Mean {dim}: {v}")

        print(f"  Refusals: {row['n_refusals']} ({row['pct_refusal']}%)")
        print(f"  Degenerate: {row['n_degenerate']} ({row['pct_degenerate']}%)")
        print(f"  Parse errors: {row['n_errors']}")

    if per_book:
        print(f"\n--- Per-book breakdown ---")
        for row in per_book:
            dims = {k: v for k, v in row.items() if k.startswith("mean_") and k != "mean_utility"}
            dim_str = ", ".join(f"{k.replace('mean_', '')}={v}" for k, v in dims.items())
            print(f"  {row['sample_type']:>10s} | {row['book']:>20s} | "
                  f"utility={row['mean_utility']} | {dim_str} | "
                  f"n={row['n']}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(samples: List[UtilitySample], model: str):
    """Print estimated API cost."""
    avg_input = 350
    avg_output = 100
    n = len(samples)

    # Claude Sonnet pricing
    input_rate = 3.0  # per MTok
    output_rate = 15.0  # per MTok

    input_cost = n * avg_input * input_rate / 1_000_000
    output_cost = n * avg_output * output_rate / 1_000_000
    total = input_cost + output_cost

    print(f"\nCost estimate ({model}, {n} samples):")
    print(f"  Input:  ${input_cost:.4f} (~{n * avg_input:,} tokens)")
    print(f"  Output: ${output_cost:.4f} (~{n * avg_output:,} tokens)")
    print(f"  Total:  ${total:.4f}")

    return total


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_existing_results(output_path: str):
    """Load already-judged results for resume."""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {r["id"] for r in existing if "scores" in r and not r["scores"].get("parse_error")}
        return done_ids, existing
    return set(), []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    # 1. Load all samples
    all_samples = []
    for input_path, stype in zip(args.input, args.type):
        samples = load_samples(input_path, stype)
        print(f"Loaded {len(samples)} {stype} samples from {input_path}")
        all_samples.append(samples)

    samples_flat = [s for group in all_samples for s in group]
    print(f"Total: {len(samples_flat)} samples")

    # 2. Pre-screen summary
    n_refusals = sum(1 for s in samples_flat if s.flags["is_refusal"])
    n_degenerate = sum(1 for s in samples_flat if s.flags["is_degenerate"])
    n_empty = sum(1 for s in samples_flat if s.flags["is_empty"])
    print(f"Pre-screen: {n_refusals} refusals, {n_degenerate} degenerate, {n_empty} empty")

    # 3. Cost estimate
    estimate_cost(samples_flat, args.model)

    if args.dry_run:
        print("\n--- DRY RUN: Example prompts ---\n")
        for s in samples_flat[:2]:
            print(f"[{s.sample_type}] {s.id}")
            print(build_prompt(s))
            print("\n" + "-" * 60 + "\n")
        print("Dry run complete. No API calls made.")
        return

    # 4. Resume support
    done_ids = set()
    existing_results = []
    if args.resume:
        done_ids, existing_data = load_existing_results(args.output)
        # Reconstruct UtilitySample from existing data
        for item in existing_data:
            if item["id"] in done_ids:
                s = UtilitySample(
                    id=item["id"], book=item["book"],
                    question=item["question"], reference="",
                    generated=item["generated"],
                    sample_type=item["sample_type"],
                    flags=item["flags"], scores=item["scores"],
                    utility_score=item["utility_score"],
                )
                existing_results.append(s)
        print(f"Resuming: {len(done_ids)} already judged, "
              f"{len(samples_flat) - len(done_ids)} remaining")

    pending = [s for s in samples_flat if s.id not in done_ids]

    if not pending:
        print("All samples already judged. Nothing to do.")
        results = existing_results
    else:
        # 5. Judge via Anthropic API
        client = AsyncAnthropic()
        semaphore = asyncio.Semaphore(args.max_concurrent)
        rate_limiter = RateLimiter(args.rate_limit)

        results = list(existing_results)
        completed = 0
        total = len(pending)

        # Process in batches for incremental save
        tasks = []
        for s in pending:
            task = asyncio.create_task(
                judge_sample(client, s, args.model, semaphore, rate_limiter)
            )
            tasks.append(task)

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if completed % 10 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")

            # Incremental save every 20 samples
            if completed % 20 == 0:
                save_results(results, args.output)

    # 6. Final save and summary
    save_results(results, args.output)
    print_summary(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate utility of model responses using Claude as AI judge."
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="JSON file(s) with model outputs (must have 'generated' field)")
    parser.add_argument("--type", nargs="+", required=True,
                        choices=["nonliteral", "qa"],
                        help="Sample type for each input file: nonliteral or qa")
    parser.add_argument("--output", type=str, default="outputs/utility_results.json",
                        help="Output JSON path (default: outputs/utility_results.json)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929",
                        help="Anthropic model for judging (default: claude-sonnet-4-5-20250929)")
    parser.add_argument("--max_concurrent", type=int, default=5,
                        help="Max concurrent API calls (default: 5)")
    parser.add_argument("--rate_limit", type=int, default=50,
                        help="Requests per minute (default: 50)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompts + cost estimate without calling API")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial output (skip already-judged IDs)")

    args = parser.parse_args()

    if len(args.input) != len(args.type):
        parser.error("Number of --input files must match number of --type values")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
