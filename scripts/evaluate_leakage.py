"""
Leakage Evaluation
==================
Score generated outputs against reference passages using five complementary
metrics. Each metric is independently selectable via --metrics.

  1. rouge_l       - ROUGE-L (longest common subsequence F-measure)
  2. token_overlap - Unordered token intersection ratio
  3. lcs           - Block-based near-verbatim recall (NV-Recall) from
                     "Extracting books from production language models"
  4. phase1_lcs    - Longest contiguous common substring, normalised by reference length
  5. embedding     - Calibrated cosine similarity, flagged if p < alpha against the
                     null distribution from calibrate_thresholds.py

Usage:
    python scripts/evaluate_leakage.py --input outputs/sft_baseline/literal_outputs.json --metrics rouge_l lcs
    python scripts/evaluate_leakage.py --input outputs/sft_baseline/nonliteral_outputs.json --metrics embedding
    python scripts/evaluate_leakage.py --input outputs/sft_baseline/literal_outputs.json --metrics all
    python scripts/evaluate_leakage.py --input outputs/sft_baseline/literal_outputs.json --metrics all --thresholds outputs/calibration/thresholds.json
"""

import argparse
import csv
import json
import os
import difflib

# ---------------------------------------------------------------------------
# Metric 1: ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(reference, generated):
    """Compute ROUGE-L F-measure between reference and generated text."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# Metric 2: Token Overlap
# ---------------------------------------------------------------------------

def compute_token_overlap(reference, generated):
    """Compute unordered token intersection ratio.

    token_overlap = |ref_tokens ∩ gen_tokens| / |ref_tokens|
    """
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())

    if not ref_tokens:
        return 0.0

    intersection = ref_tokens & gen_tokens
    return len(intersection) / len(ref_tokens)


# ---------------------------------------------------------------------------
# Metric 3: Block-based Longest Common Substring (nv-recall)
#
# Following "Extracting books from production language models" (Ahmed et al.):
#   1. Identify verbatim matching blocks via greedy LCS (difflib.SequenceMatcher)
#   2. Merge 1: stitch blocks separated by tiny gaps (gap <= 2, alignment <= 1)
#   3. Filter 1: drop blocks shorter than min_len_1 words
#   4. Merge 2: passage-level merge (gap <= 10, alignment <= 3)
#   5. Filter 2: drop blocks shorter than min_len_2 words
#   6. nv-recall = total matched words / |reference|
#
# Thresholds adapted for our scale (200-token generations vs 200-token references):
#   - Paper uses l(1)=20, l(2)=100 for full-book extraction
#   - We use l(1)=5, l(2)=10 since our texts are much shorter
# ---------------------------------------------------------------------------

def _get_matching_blocks(ref_words, gen_words):
    """Find verbatim matching blocks using difflib (greedy LCS approximation)."""
    sm = difflib.SequenceMatcher(None, ref_words, gen_words, autojunk=False)
    blocks = []
    for block in sm.get_matching_blocks():
        if block.size > 0:
            blocks.append((block.a, block.b, block.size))
    return blocks


def _merge_blocks(blocks, max_gap, max_align_diff):
    """Merge nearby blocks if gap and alignment difference are within thresholds.

    gap = distance between end of block k and start of block k+1 in reference
    align_diff = |gap_in_ref - gap_in_gen| (how aligned the blocks are)
    """
    if not blocks:
        return []

    merged = [blocks[0]]
    for i_ref, j_gen, length in blocks[1:]:
        prev_i, prev_j, prev_len = merged[-1]
        gap_ref = i_ref - (prev_i + prev_len)
        gap_gen = j_gen - (prev_j + prev_len)
        align_diff = abs(gap_ref - gap_gen)

        if 0 <= gap_ref <= max_gap and 0 <= gap_gen <= max_gap and align_diff <= max_align_diff:
            # Merge: extend previous block to cover through current block
            # Only count verbatim-matched words (not gap words)
            new_len = prev_len + length
            merged[-1] = (prev_i, prev_j, new_len)
        else:
            merged.append((i_ref, j_gen, length))

    return merged


def _filter_blocks(blocks, min_length):
    """Remove blocks shorter than min_length."""
    return [(i, j, m) for i, j, m in blocks if m >= min_length]


def compute_lcs_nv_recall(reference, generated, min_len_1=5, min_len_2=10):
    """Compute block-based nv-recall following the paper's Algorithm 1.

    Returns dict with:
        - nv_recall: proportion of reference covered by near-verbatim blocks
        - total_matched: total matched words
        - num_blocks: number of final blocks
        - longest_block: length of longest final block (in words)
        - ref_length: length of reference in words
    """
    ref_words = reference.split()
    gen_words = generated.split()

    if not ref_words or not gen_words:
        return {
            "nv_recall": 0.0,
            "total_matched": 0,
            "num_blocks": 0,
            "longest_block": 0,
            "ref_length": len(ref_words),
        }

    # Step 1: Identify verbatim matching blocks
    blocks = _get_matching_blocks(ref_words, gen_words)

    # Step 2: Merge 1 — stitch very short gaps
    blocks = _merge_blocks(blocks, max_gap=2, max_align_diff=1)

    # Step 3: Filter 1 — remove short blocks
    blocks = _filter_blocks(blocks, min_len_1)

    # Step 4: Merge 2 — passage-level consolidation
    blocks = _merge_blocks(blocks, max_gap=10, max_align_diff=3)

    # Step 5: Filter 2 — retain sufficiently long blocks
    blocks = _filter_blocks(blocks, min_len_2)

    # Compute metrics
    total_matched = sum(m for _, _, m in blocks)
    longest_block = max((m for _, _, m in blocks), default=0)
    nv_recall = total_matched / len(ref_words) if ref_words else 0.0

    return {
        "nv_recall": nv_recall,
        "total_matched": total_matched,
        "num_blocks": len(blocks),
        "longest_block": longest_block,
        "ref_length": len(ref_words),
    }


# ---------------------------------------------------------------------------
# Metric 4: Phase-1 LCS (normalized longest common substring)
#
# From the paper: s(T,R) = longest(T,R) / |T|
# Measures the longest contiguous verbatim substring as a fraction of reference.
# ---------------------------------------------------------------------------

def _longest_common_substring_length(tokens_a, tokens_b):
    """Compute length of longest contiguous common substring using DP."""
    n = len(tokens_a)
    m = len(tokens_b)
    # Use 1D DP to save memory
    prev = [0] * (m + 1)
    longest = 0
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > longest:
                    longest = curr[j]
        prev = curr
    return longest


def compute_phase1_similarity(reference, generated):
    """Compute s(T,R) = longest(T,R) / |T| from the paper."""
    ref_words = reference.split()
    gen_words = generated.split()
    if not ref_words:
        return 0.0
    longest = _longest_common_substring_length(ref_words, gen_words)
    return longest / len(ref_words)


# ---------------------------------------------------------------------------
# Metric 5: Calibrated Embedding Similarity
#
# Replaces BERTScore with the advisor's approach:
#   1. Compute embedding cosine similarity between generated and reference
#   2. Compare against the null distribution (from calibration experiment)
#   3. Compute z-score and p-value
#   4. Flag as copying if p-value < alpha (default 0.001)
#
# Requires: thresholds.json from calibrate_thresholds.py
# ---------------------------------------------------------------------------

_embed_model = None
_calibration = None


def _load_embed_model(model_name="all-MiniLM-L6-v2"):
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def load_calibration(thresholds_path):
    """Load calibration thresholds for embedding similarity."""
    global _calibration
    if _calibration is None:
        with open(thresholds_path) as f:
            _calibration = json.load(f)
    return _calibration


def compute_embedding_similarity(reference, generated):
    """Compute calibrated embedding similarity with z-score and p-value.

    Returns dict with:
        - embedding_sim: raw cosine similarity
        - embedding_z: z-score relative to null distribution
        - embedding_pvalue: one-tailed p-value
        - embedding_flagged: True if p-value < alpha (copying detected)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import stats

    model = _load_embed_model()
    emb_ref = model.encode([reference])
    emb_gen = model.encode([generated])
    sim = float(cosine_similarity(emb_ref, emb_gen)[0][0])

    # If calibration is loaded, compute z-score and p-value
    if _calibration and "embedding_sim" in _calibration:
        cal = _calibration["embedding_sim"]
        null_mean = cal["null_mean"]
        null_std = cal["null_std"]
        alpha = cal["alpha"]

        if null_std > 0:
            z = (sim - null_mean) / null_std
            pvalue = 1.0 - stats.norm.cdf(z)  # one-tailed
        else:
            z = 0.0
            pvalue = 1.0

        return {
            "embedding_sim": sim,
            "embedding_z": round(z, 4),
            "embedding_pvalue": round(pvalue, 6),
            "embedding_flagged": int(pvalue < alpha),
        }
    else:
        return {"embedding_sim": sim}


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

METRIC_FUNCTIONS = {
    "rouge_l": compute_rouge_l,
    "token_overlap": compute_token_overlap,
    "lcs": None,  # handled separately (returns dict)
    "phase1_lcs": compute_phase1_similarity,
    "embedding": None,  # handled separately (returns dict)
}

ALL_METRICS = list(METRIC_FUNCTIONS.keys())


def evaluate_sample(reference, generated, metrics):
    """Compute selected metrics for a single sample."""
    results = {}
    for metric in metrics:
        if metric == "lcs":
            lcs_result = compute_lcs_nv_recall(reference, generated)
            results["lcs_nv_recall"] = lcs_result["nv_recall"]
            results["lcs_total_matched"] = lcs_result["total_matched"]
            results["lcs_num_blocks"] = lcs_result["num_blocks"]
            results["lcs_longest_block"] = lcs_result["longest_block"]
            results["lcs_ref_length"] = lcs_result["ref_length"]
        elif metric == "embedding":
            emb_result = compute_embedding_similarity(reference, generated)
            results.update(emb_result)
        else:
            fn = METRIC_FUNCTIONS[metric]
            results[metric] = fn(reference, generated)
    return results


def main(args):
    # Parse metrics
    if "all" in args.metrics:
        metrics = ALL_METRICS
    else:
        metrics = args.metrics
        for m in metrics:
            if m not in ALL_METRICS:
                print(f"Unknown metric: {m}. Available: {ALL_METRICS}")
                return

    print(f"Input: {args.input}")
    print(f"Metrics: {metrics}")

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect format: literal (prefix/reference/generated) or nonliteral (question/reference/generated)
    sample = data[0]
    if "prefix" in sample:
        ref_key = "reference"
        prompt_key = "prefix"
        print(f"Detected literal format ({len(data)} samples)")
    elif "question" in sample:
        ref_key = "reference"
        prompt_key = "question"
        print(f"Detected non-literal format ({len(data)} samples)")
    else:
        print("Unknown data format. Expected 'prefix' or 'question' key.")
        return

    # Evaluate each sample
    all_results = []
    for item in data:
        reference = item.get(ref_key, "")
        generated = item.get("generated", "")

        if not generated:
            print(f"  [skip] {item.get('id', '?')} — no generated text")
            continue

        scores = evaluate_sample(reference, generated, metrics)
        result = {"id": item.get("id", ""), "book": item.get("book", "")}
        result.update(scores)
        all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results ({len(all_results)} samples)")
    print(f"{'='*60}")

    for metric in metrics:
        if metric == "lcs":
            values = [r["lcs_nv_recall"] for r in all_results]
            longest_values = [r["lcs_longest_block"] for r in all_results]
            if values:
                avg = sum(values) / len(values)
                max_longest = max(longest_values)
                avg_longest = sum(longest_values) / len(longest_values)
                print(f"  LCS nv-recall:     avg={avg:.4f}, max_longest_block={max_longest}")
                print(f"  LCS longest block: avg={avg_longest:.1f} words")
        elif metric == "embedding":
            sim_values = [r["embedding_sim"] for r in all_results if "embedding_sim" in r]
            if sim_values:
                avg_sim = sum(sim_values) / len(sim_values)
                max_sim = max(sim_values)
                min_sim = min(sim_values)
                print(f"  embedding_sim: avg={avg_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")
            if "embedding_z" in all_results[0]:
                z_values = [r["embedding_z"] for r in all_results]
                avg_z = sum(z_values) / len(z_values)
                print(f"  embedding_z:   avg={avg_z:.4f}")
            if "embedding_flagged" in all_results[0]:
                flagged = sum(r["embedding_flagged"] for r in all_results)
                total = len(all_results)
                print(f"  embedding_flagged: {flagged}/{total} ({flagged/total*100:.1f}%) samples flagged as copying")
        else:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                avg = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                print(f"  {metric}: avg={avg:.4f}, min={min_val:.4f}, max={max_val:.4f}")

    # Save to CSV
    if args.output and all_results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fieldnames = list(all_results[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved to {args.output}")

    # Also save enriched JSON (original data + scores)
    if args.output:
        json_output = args.output.replace(".csv", ".json")
        for item, result in zip(data, all_results):
            item.update(result)
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved enriched JSON to {json_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to model outputs JSON (literal or non-literal)")
    parser.add_argument("--output", type=str, default="outputs/leakage_results.csv",
                        help="Path to save results CSV")
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        help="Metrics to compute: rouge_l, token_overlap, lcs, phase1_lcs, embedding, all")
    parser.add_argument("--thresholds", type=str,
                        default="outputs/calibration/thresholds.json",
                        help="Path to calibration thresholds (for embedding metric)")
    args = parser.parse_args()

    # Load calibration if embedding metric is requested
    if "all" in args.metrics or "embedding" in args.metrics:
        if os.path.exists(args.thresholds):
            load_calibration(args.thresholds)
            print(f"Loaded calibration thresholds from {args.thresholds}")
        else:
            print(f"WARNING: Thresholds file not found at {args.thresholds}")
            print("  Embedding similarity will be computed without calibration (no z-score/p-value)")

    main(args)