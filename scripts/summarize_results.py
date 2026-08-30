"""
Summarize Leakage Results
=========================
Read one or more leakage_results CSV files (output of evaluate_leakage.py),
compute summary statistics across samples, and optionally generate plots.

Usage:
    python scripts/summarize_results.py --input outputs/sft_baseline/leakage_literal.csv
    python scripts/summarize_results.py --input outputs/sft_baseline/leakage_literal.csv outputs/sft_baseline/leakage_nonliteral.csv
    python scripts/summarize_results.py --input outputs/sft_baseline/leakage_literal.csv --plot
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(paths):
    """Load results from one or more CSV files."""
    all_rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Convert numeric fields
            for row in rows:
                for key in row:
                    if key in ("id", "book"):
                        continue
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            all_rows.extend(rows)
            print(f"Loaded {len(rows)} rows from {path}")
    return all_rows


def compute_stats(values):
    """Compute summary statistics for a list of values."""
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "min": min(values),
        "max": max(values),
        "std": (sum((x - mean) ** 2 for x in values) / n) ** 0.5,
    }


def summarize(rows):
    """Print summary tables."""
    # Detect which metrics are present
    metric_keys = [k for k in rows[0].keys() if k not in ("id", "book")]

    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY ({len(rows)} samples)")
    print(f"{'='*70}")

    for key in metric_keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        stats = compute_stats(values)
        print(f"\n  {key}:")
        print(f"    mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
              f"min={stats['min']:.4f}  max={stats['max']:.4f}  std={stats['std']:.4f}")

    # Per-book breakdown
    books = defaultdict(list)
    for row in rows:
        books[row.get("book", "unknown")].append(row)

    if len(books) > 1:
        print(f"\n{'='*70}")
        print(f"PER-BOOK BREAKDOWN")
        print(f"{'='*70}")

        # Print as table
        header = f"{'Book':<25}"
        for key in metric_keys:
            if key in ("lcs_ref_length",):
                continue
            header += f"  {key:>12}"
        print(f"\n{header}")
        print("-" * len(header))

        for book_name in sorted(books.keys()):
            book_rows = books[book_name]
            line = f"{book_name:<25}"
            for key in metric_keys:
                if key in ("lcs_ref_length",):
                    continue
                values = [r[key] for r in book_rows if isinstance(r.get(key), (int, float))]
                if values:
                    avg = sum(values) / len(values)
                    line += f"  {avg:>12.4f}"
                else:
                    line += f"  {'N/A':>12}"
            print(line)

    # Extraction cases (nv-recall > threshold)
    if any("lcs_nv_recall" in r for r in rows):
        nv_values = [r["lcs_nv_recall"] for r in rows if isinstance(r.get("lcs_nv_recall"), (int, float))]
        if nv_values:
            thresholds = [0.1, 0.3, 0.5, 0.7]
            print(f"\n{'='*70}")
            print(f"EXTRACTION CASES (nv-recall thresholds)")
            print(f"{'='*70}")
            for t in thresholds:
                count = sum(1 for v in nv_values if v >= t)
                print(f"  nv-recall >= {t:.1f}: {count}/{len(nv_values)} ({100*count/len(nv_values):.1f}%)")


def plot_results(rows, output_path):
    """Generate a comprehensive results figure."""
    books_data = defaultdict(list)
    for r in rows:
        books_data[r.get("book", "unknown")].append(r)

    book_names = sorted(books_data.keys())
    book_labels = [b.replace("_", " ").title() for b in book_names]

    COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]
    book_colors = {bn: COLORS[i % len(COLORS)] for i, bn in enumerate(book_names)}

    # Detect available metrics (score-like, 0-1 range)
    score_metrics = []
    for key in ["rouge_l", "lcs_nv_recall", "phase1_lcs", "token_overlap"]:
        if any(isinstance(r.get(key), (int, float)) for r in rows):
            score_metrics.append(key)

    metric_labels = {
        "rouge_l": "ROUGE-L",
        "lcs_nv_recall": "NV-Recall",
        "phase1_lcs": "Phase-1 LCS",
        "token_overlap": "Token Overlap",
    }
    metric_colors = {
        "rouge_l": "#e74c3c",
        "lcs_nv_recall": "#3498db",
        "phase1_lcs": "#2ecc71",
        "token_overlap": "#f39c12",
    }

    n_metrics = len(score_metrics)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "CopyShield: Literal Leakage Evaluation Results\n"
        "LLaMA-3.1-8B + QLoRA (r=64, \u03b1=128, 20 epochs) \u00b7 "
        f"{len(rows)} samples \u00b7 150-token prefix \u00b7 200-token reference",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Row 1: Histograms for each metric
    for idx, key in enumerate(score_metrics):
        ax = fig.add_subplot(3, max(n_metrics, 4), idx + 1)
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        ax.hist(values, bins=20, color=metric_colors.get(key, "#888"),
                alpha=0.8, edgecolor="white")
        mean_val = np.mean(values)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_val:.3f}")
        ax.set_title(metric_labels.get(key, key), fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.2)

    # Row 2 left: Per-book bar chart
    ax = fig.add_subplot(3, 2, 3)
    x = np.arange(len(book_names))
    width = 0.8 / max(n_metrics, 1)
    for i, key in enumerate(score_metrics):
        means = []
        for bn in book_names:
            vals = [r[key] for r in books_data[bn] if isinstance(r.get(key), (int, float))]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, means, width,
               label=metric_labels.get(key, key),
               color=metric_colors.get(key, "#888"), alpha=0.85)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(book_labels, fontsize=9)
    ax.set_ylabel("Mean Score")
    ax.set_title("Per-Book Metric Comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    # Row 2 right: Extraction success rates
    ax = fig.add_subplot(3, 2, 4)
    has_nv = any(isinstance(r.get("lcs_nv_recall"), (int, float)) for r in rows)
    if has_nv:
        thresholds = [0.1, 0.3, 0.5, 0.7]
        th_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
        bar_w = 0.8 / len(thresholds)
        for i, bn in enumerate(book_names):
            nv_vals = [r["lcs_nv_recall"] for r in books_data[bn]
                       if isinstance(r.get("lcs_nv_recall"), (int, float))]
            n = len(nv_vals)
            for j, (t, tc) in enumerate(zip(thresholds, th_colors)):
                rate = sum(1 for v in nv_vals if v >= t) / n * 100 if n else 0
                ax.bar(i + j * bar_w, rate, bar_w, color=tc, alpha=0.85,
                       label=f"\u2265{t}" if i == 0 else "")
        ax.set_xticks(range(len(book_names)))
        ax.set_xticklabels(book_labels, fontsize=9)
        ax.set_ylabel("% of Prompts")
        ax.set_title("Extraction Success Rate (NV-Recall Thresholds)",
                      fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    # Row 3 left: Box plots per book for NV-recall
    ax = fig.add_subplot(3, 2, 5)
    if has_nv:
        box_data = []
        for bn in book_names:
            vals = [r["lcs_nv_recall"] for r in books_data[bn]
                    if isinstance(r.get("lcs_nv_recall"), (int, float))]
            box_data.append(vals)
        bp = ax.boxplot(box_data, labels=book_labels, patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
        for patch, bn in zip(bp["boxes"], book_names):
            patch.set_facecolor(book_colors[bn])
            patch.set_alpha(0.6)
        ax.set_ylabel("NV-Recall")
        ax.set_title("NV-Recall Distribution Per Book", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    # Row 3 right: Phase-1 LCS vs NV-Recall scatter
    ax = fig.add_subplot(3, 2, 6)
    has_p1 = any(isinstance(r.get("phase1_lcs"), (int, float)) for r in rows)
    if has_nv and has_p1:
        for bn in book_names:
            nv = [r["lcs_nv_recall"] for r in books_data[bn]
                  if isinstance(r.get("lcs_nv_recall"), (int, float))]
            p1 = [r["phase1_lcs"] for r in books_data[bn]
                  if isinstance(r.get("phase1_lcs"), (int, float))]
            ax.scatter(p1, nv, color=book_colors[bn], alpha=0.6, s=30,
                       label=bn.replace("_", " ").title())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Phase-1 LCS")
        ax.set_ylabel("NV-Recall")
        ax.set_title("Phase-1 LCS vs NV-Recall (per sample)",
                      fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.2)

    # Summary text box
    total = len(rows)
    lines = [
        f"SUMMARY STATISTICS",
        f"{chr(9472) * 40}",
        f"Total samples: {total}",
        f"Prefix: 150 tokens | Reference: 200 tokens",
        f"{chr(9472) * 40}",
    ]
    for key in score_metrics:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            lines.append(f"{metric_labels.get(key, key):14s} mean={np.mean(vals):.3f}  max={max(vals):.3f}")

    if has_nv:
        nv_vals = [r["lcs_nv_recall"] for r in rows
                   if isinstance(r.get("lcs_nv_recall"), (int, float))]
        lines.append(f"{chr(9472) * 40}")
        lines.append("Extraction rates (NV-Recall):")
        for t in [0.1, 0.3, 0.5, 0.7]:
            c = sum(1 for v in nv_vals if v >= t)
            lines.append(f"  \u2265{t}: {c:>3d}/{total} ({100*c/total:.1f}%)")

    fig.text(0.02, 0.01, "\n".join(lines), fontsize=9, fontfamily="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main(args):
    rows = load_results(args.input)
    if not rows:
        print("No results to summarize.")
        return
    summarize(rows)

    if args.plot:
        # Derive output path from first input file
        base = os.path.splitext(args.input[0])[0]
        plot_path = base + "_plot.png"
        plot_results(rows, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True,
                        help="Path(s) to leakage results CSV file(s)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate results plot image")
    args = parser.parse_args()
    main(args)
