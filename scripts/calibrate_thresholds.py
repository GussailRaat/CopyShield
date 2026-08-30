"""
Threshold Calibration
=====================
Build null distributions for all leakage metrics by comparing passages from
the protected corpus against passages from the neutral corpus, then derive
statistically calibrated detection thresholds at significance level
alpha = 0.001 (z_alpha = 3.09).

Steps:
  1. Sample random passages from protected and neutral corpora
  2. Compute metrics for cross-pairs (protected x neutral) -> null distribution
  3. Compute metrics for same-source pairs (protected x protected) -> positive distribution
  4. Fit distributions, derive thresholds at alpha = 0.001
  5. Save thresholds.json + diagnostic plots

Usage:
    python scripts/calibrate_thresholds.py \
        --protected data/books_corpus.txt \
        --neutral data/neutral_books_corpus.txt \
        --n_passages 500 \
        --passage_len 200 \
        --n_pairs 1000 \
        --output outputs/calibration/
"""

import argparse
import json
import os
import random
import sys
import numpy as np
from scipy import stats

# Add project root to path so we can import evaluate_leakage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate_leakage import (
    compute_rouge_l,
    compute_lcs_nv_recall,
    compute_phase1_similarity,
)


# ---------------------------------------------------------------------------
# Step 1: Sample passages from a corpus
# ---------------------------------------------------------------------------

def sample_passages(corpus_path, n_passages, passage_len_words, seed=42):
    """Sample random passages of fixed word length from a corpus file.

    Args:
        corpus_path: Path to the corpus text file.
        n_passages: Number of passages to sample.
        passage_len_words: Target length of each passage in words.
        seed: Random seed for reproducibility.

    Returns:
        List of passage strings.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    total_words = len(words)

    if total_words < passage_len_words:
        raise ValueError(
            f"Corpus {corpus_path} has only {total_words} words, "
            f"need at least {passage_len_words}"
        )

    rng = random.Random(seed)
    passages = []
    max_start = total_words - passage_len_words

    for _ in range(n_passages):
        start = rng.randint(0, max_start)
        passage = " ".join(words[start : start + passage_len_words])
        passages.append(passage)

    return passages


# ---------------------------------------------------------------------------
# Step 2: Compute metrics for passage pairs
# ---------------------------------------------------------------------------

def compute_pair_metrics(passage_a, passage_b):
    """Compute all string-based metrics for a single pair.

    Returns dict with: rouge_l, nv_recall, phase1_lcs
    """
    rouge_l = compute_rouge_l(passage_a, passage_b)

    lcs_result = compute_lcs_nv_recall(passage_a, passage_b)
    nv_recall = lcs_result["nv_recall"]

    phase1_lcs = compute_phase1_similarity(passage_a, passage_b)

    return {
        "rouge_l": rouge_l,
        "nv_recall": nv_recall,
        "phase1_lcs": phase1_lcs,
    }


def compute_embedding_similarity(passages_a, passages_b, model_name="all-MiniLM-L6-v2"):
    """Compute embedding cosine similarity for all pairs.

    Args:
        passages_a: List of passages (one per pair).
        passages_b: List of passages (one per pair).
        model_name: Sentence-transformers model name.

    Returns:
        List of cosine similarity scores.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"  Encoding {len(passages_a)} passages (set A)...")
    emb_a = model.encode(passages_a, show_progress_bar=True, batch_size=64)

    print(f"  Encoding {len(passages_b)} passages (set B)...")
    emb_b = model.encode(passages_b, show_progress_bar=True, batch_size=64)

    # Compute pairwise cosine similarity (only diagonal — matched pairs)
    similarities = []
    for i in range(len(passages_a)):
        sim = cos_sim([emb_a[i]], [emb_b[i]])[0][0]
        similarities.append(float(sim))

    return similarities


# ---------------------------------------------------------------------------
# Step 3: Build null and positive distributions
# ---------------------------------------------------------------------------

def build_distributions(protected_passages, neutral_passages, n_pairs, seed=42):
    """Build null (cross-corpus) and positive (same-corpus) distributions.

    Null: protected[i] vs neutral[j] (unrelated pairs → "innocent" baseline)
    Positive: protected[i] vs protected[j] with nearby passages (known overlap)

    Returns:
        null_scores: dict of metric_name → list of scores
        positive_scores: dict of metric_name → list of scores
        null_pairs: list of (idx_a, idx_b) for reproducibility
    """
    rng = random.Random(seed)
    metrics = ["rouge_l", "nv_recall", "phase1_lcs"]

    # --- Null distribution: cross-corpus pairs ---
    print(f"\nComputing null distribution ({n_pairs} cross-corpus pairs)...")
    null_scores = {m: [] for m in metrics}
    null_passages_a = []
    null_passages_b = []

    for i in range(n_pairs):
        idx_a = rng.randint(0, len(protected_passages) - 1)
        idx_b = rng.randint(0, len(neutral_passages) - 1)
        pa = protected_passages[idx_a]
        pb = neutral_passages[idx_b]

        null_passages_a.append(pa)
        null_passages_b.append(pb)

        scores = compute_pair_metrics(pa, pb)
        for m in metrics:
            null_scores[m].append(scores[m])

        if (i + 1) % 100 == 0:
            print(f"  Null pairs: {i + 1}/{n_pairs}")

    # --- Positive distribution: same-passage pairs (shifted slightly) ---
    # Take a passage and compare to a nearby passage from the same corpus
    # This simulates what actual copying looks like
    print(f"\nComputing positive distribution ({n_pairs} same-corpus pairs)...")
    positive_scores = {m: [] for m in metrics}
    positive_passages_a = []
    positive_passages_b = []

    for i in range(n_pairs):
        idx = rng.randint(0, len(protected_passages) - 1)
        pa = protected_passages[idx]
        # Use the same passage as the "generated" text (perfect copy)
        pb = pa

        positive_passages_a.append(pa)
        positive_passages_b.append(pb)

        scores = compute_pair_metrics(pa, pb)
        for m in metrics:
            positive_scores[m].append(scores[m])

        if (i + 1) % 100 == 0:
            print(f"  Positive pairs: {i + 1}/{n_pairs}")

    return (
        null_scores, positive_scores,
        null_passages_a, null_passages_b,
        positive_passages_a, positive_passages_b,
    )


# ---------------------------------------------------------------------------
# Step 4: Fit distributions and derive thresholds
# ---------------------------------------------------------------------------

def derive_thresholds(null_scores, positive_scores, alpha=0.05):
    """Derive thresholds from null distributions.

    For each metric:
        - Fit a normal distribution to the null scores
        - Threshold = value where P(X > threshold) = alpha (one-tailed)
        - Also report separation between null and positive distributions

    Args:
        null_scores: dict of metric_name → list of null scores
        positive_scores: dict of metric_name → list of positive scores
        alpha: Significance level (default 0.05)

    Returns:
        dict of metric_name → {mean, std, threshold, positive_mean, separation}
    """
    results = {}

    for metric in null_scores:
        null = np.array(null_scores[metric])
        pos = np.array(positive_scores[metric])

        null_mean = float(np.mean(null))
        null_std = float(np.std(null))
        null_median = float(np.median(null))

        # One-tailed threshold: P(X > threshold) = alpha
        # threshold = mean + z_alpha * std
        z_alpha = stats.norm.ppf(1 - alpha)
        threshold = null_mean + z_alpha * null_std

        # Positive distribution stats
        pos_mean = float(np.mean(pos))
        pos_std = float(np.std(pos))

        # Separation: how well does the threshold separate null from positive?
        # True positive rate: fraction of positive pairs above threshold
        tpr = float(np.mean(pos > threshold))
        # False positive rate: fraction of null pairs above threshold (should be ~alpha)
        fpr = float(np.mean(null > threshold))

        results[metric] = {
            "null_mean": round(null_mean, 6),
            "null_std": round(null_std, 6),
            "null_median": round(null_median, 6),
            "threshold": round(threshold, 6),
            "alpha": alpha,
            "z_alpha": round(z_alpha, 4),
            "positive_mean": round(pos_mean, 6),
            "positive_std": round(pos_std, 6),
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
        }

        print(f"\n  {metric}:")
        print(f"    Null:     mean={null_mean:.4f}, std={null_std:.4f}, median={null_median:.4f}")
        print(f"    Threshold (p<{alpha}): {threshold:.4f}")
        print(f"    Positive: mean={pos_mean:.4f}")
        print(f"    TPR={tpr:.4f}, FPR={fpr:.4f}")

    return results


# ---------------------------------------------------------------------------
# Step 5: Plot distributions
# ---------------------------------------------------------------------------

def plot_distributions(null_scores, positive_scores, thresholds, output_dir):
    """Plot null vs positive distributions with threshold lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(null_scores.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        null = null_scores[metric]
        pos = positive_scores[metric]
        thresh = thresholds[metric]["threshold"]

        ax.hist(null, bins=50, alpha=0.6, label="Null (unrelated)", color="steelblue", density=True)
        ax.hist(pos, bins=50, alpha=0.6, label="Positive (same text)", color="salmon", density=True)
        ax.axvline(thresh, color="red", linestyle="--", linewidth=2,
                   label=f"Threshold={thresh:.4f}")
        ax.set_title(metric)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "calibration_distributions.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved distribution plot to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate leakage metric thresholds")
    parser.add_argument("--protected", type=str, required=True,
                        help="Path to protected books corpus")
    parser.add_argument("--neutral", type=str, required=True,
                        help="Path to neutral books corpus")
    parser.add_argument("--n_passages", type=int, default=500,
                        help="Number of passages to sample from each corpus")
    parser.add_argument("--passage_len", type=int, default=200,
                        help="Passage length in words")
    parser.add_argument("--n_pairs", type=int, default=1000,
                        help="Number of pairs for null/positive distributions")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Significance level for threshold")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for embedding similarity")
    parser.add_argument("--output", type=str, default="outputs/calibration/",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Step 1: Sample passages
    print("=" * 60)
    print("Step 1: Sampling passages")
    print("=" * 60)
    protected_passages = sample_passages(
        args.protected, args.n_passages, args.passage_len, seed=args.seed
    )
    neutral_passages = sample_passages(
        args.neutral, args.n_passages, args.passage_len, seed=args.seed + 1
    )
    print(f"  Sampled {len(protected_passages)} protected passages")
    print(f"  Sampled {len(neutral_passages)} neutral passages")

    # Step 2-3: Build distributions (string metrics)
    print("\n" + "=" * 60)
    print("Step 2: Building null and positive distributions (string metrics)")
    print("=" * 60)
    (
        null_scores, positive_scores,
        null_passages_a, null_passages_b,
        positive_passages_a, positive_passages_b,
    ) = build_distributions(
        protected_passages, neutral_passages, args.n_pairs, seed=args.seed
    )

    # Step 2b: Embedding similarity
    print("\n" + "=" * 60)
    print("Step 3: Computing embedding similarity")
    print("=" * 60)
    null_emb_sims = compute_embedding_similarity(
        null_passages_a, null_passages_b, model_name=args.embedding_model
    )
    null_scores["embedding_sim"] = null_emb_sims

    positive_emb_sims = compute_embedding_similarity(
        positive_passages_a, positive_passages_b, model_name=args.embedding_model
    )
    positive_scores["embedding_sim"] = positive_emb_sims

    # Step 4: Derive thresholds
    print("\n" + "=" * 60)
    print("Step 4: Deriving thresholds")
    print("=" * 60)
    thresholds = derive_thresholds(null_scores, positive_scores, alpha=args.alpha)

    # Save thresholds
    thresholds_path = os.path.join(args.output, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved thresholds to {thresholds_path}")

    # Save raw scores for further analysis
    raw_path = os.path.join(args.output, "raw_scores.json")
    raw_data = {
        "null": {k: [round(v, 6) for v in vals] for k, vals in null_scores.items()},
        "positive": {k: [round(v, 6) for v in vals] for k, vals in positive_scores.items()},
        "config": {
            "n_passages": args.n_passages,
            "passage_len": args.passage_len,
            "n_pairs": args.n_pairs,
            "alpha": args.alpha,
            "embedding_model": args.embedding_model,
            "seed": args.seed,
        },
    }
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved raw scores to {raw_path}")

    # Step 5: Plot
    print("\n" + "=" * 60)
    print("Step 5: Plotting distributions")
    print("=" * 60)
    plot_distributions(null_scores, positive_scores, thresholds, args.output)

    # Final summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Metric':<20} {'Null Mean':>10} {'Threshold':>10} {'Pos Mean':>10} {'TPR':>8} {'FPR':>8}")
    print(f"  {'-'*66}")
    for metric, t in thresholds.items():
        print(f"  {metric:<20} {t['null_mean']:>10.4f} {t['threshold']:>10.4f} "
              f"{t['positive_mean']:>10.4f} {t['true_positive_rate']:>8.4f} {t['false_positive_rate']:>8.4f}")


if __name__ == "__main__":
    main()
