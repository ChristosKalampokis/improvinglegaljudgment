"""Exploratory Data Analysis for CaseHOLD dataset."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from data_loader import load_casehold


def compute_stats(ds):
    """Compute basic statistics for each split."""
    stats = {}
    for split_name in ["train", "validation", "test"]:
        split = ds[split_name]
        context_lengths = [len(ex["context"].split()) for ex in split]
        holding_lengths = []
        for ex in split:
            for h in ex["endings"]:
                holding_lengths.append(len(h.split()))

        stats[split_name] = {
            "num_samples": len(split),
            "context_words_mean": np.mean(context_lengths),
            "context_words_median": np.median(context_lengths),
            "context_words_max": np.max(context_lengths),
            "context_words_min": np.min(context_lengths),
            "holding_words_mean": np.mean(holding_lengths),
            "holding_words_median": np.median(holding_lengths),
            "holding_words_max": np.max(holding_lengths),
        }
    return stats


def label_distribution(ds):
    """Check how labels (correct holding index 0-4) are distributed."""
    dist = {}
    for split_name in ["train", "validation", "test"]:
        labels = ds[split_name]["label"]
        counts = Counter(labels)
        dist[split_name] = {k: counts[k] for k in sorted(counts.keys())}
    return dist


def plot_context_length_distribution(ds, results_dir):
    """Histogram of context lengths (word count) per split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, split_name in zip(axes, ["train", "validation", "test"]):
        lengths = [len(ex["context"].split()) for ex in ds[split_name]]
        ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"{split_name} (n={len(lengths)})")
        ax.set_xlabel("Words in context")
        ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"mean={np.mean(lengths):.0f}")
        ax.legend()
    axes[0].set_ylabel("Count")
    plt.suptitle("Context Length Distribution (word count)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "context_length_distribution.png"), dpi=150)
    plt.close()


def plot_holding_length_distribution(ds, results_dir):
    """Histogram of holding lengths (word count)."""
    all_lengths = []
    for ex in ds["train"]:
        for h in ex["endings"]:
            all_lengths.append(len(h.split()))

    plt.figure(figsize=(8, 4))
    plt.hist(all_lengths, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Words in holding")
    plt.ylabel("Count")
    plt.title("Holding Length Distribution (train set)")
    plt.axvline(np.mean(all_lengths), color="red", linestyle="--", label=f"mean={np.mean(all_lengths):.0f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "holding_length_distribution.png"), dpi=150)
    plt.close()


def plot_label_distribution(ds, results_dir):
    """Bar chart of correct label positions (0-4)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, split_name in zip(axes, ["train", "validation", "test"]):
        labels = ds[split_name]["label"]
        counts = Counter(labels)
        positions = sorted(counts.keys())
        values = [counts[p] for p in positions]
        ax.bar(positions, values, edgecolor="black", alpha=0.7)
        ax.set_title(f"{split_name}")
        ax.set_xlabel("Correct holding index")
        ax.set_xticks(positions)
    axes[0].set_ylabel("Count")
    plt.suptitle("Label Distribution (correct holding position)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "label_distribution.png"), dpi=150)
    plt.close()


def plot_token_length_distribution(ds, results_dir):
    """Estimate token counts (rough: words * 1.3) to plan for model input limits."""
    contexts = [len(ex["context"].split()) for ex in ds["train"]]
    # Rough token estimate: ~1.3 tokens per word for legal text
    token_estimates = [int(w * 1.3) for w in contexts]

    plt.figure(figsize=(8, 4))
    plt.hist(token_estimates, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Estimated tokens in context")
    plt.ylabel("Count")
    plt.title("Estimated Token Count Distribution (train set)")
    for limit, name in [(512, "BERT max"), (1024, "T5 default")]:
        plt.axvline(limit, color="red", linestyle="--", alpha=0.7, label=f"{name}={limit}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "token_length_estimates.png"), dpi=150)
    plt.close()

    over_512 = sum(1 for t in token_estimates if t > 512)
    over_1024 = sum(1 for t in token_estimates if t > 1024)
    print(f"  Contexts exceeding 512 tokens: {over_512} ({over_512/len(token_estimates)*100:.1f}%)")
    print(f"  Contexts exceeding 1024 tokens: {over_1024} ({over_1024/len(token_estimates)*100:.1f}%)")


def main():
    print("Loading CaseHOLD dataset...")
    ds = load_casehold()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Basic stats
    print("\n=== Dataset Statistics ===")
    stats = compute_stats(ds)
    for split_name, s in stats.items():
        print(f"\n  {split_name}:")
        for k, v in s.items():
            print(f"    {k}: {v:.1f}" if isinstance(v, float) else f"    {k}: {v}")

    # 2. Label distribution
    print("\n=== Label Distribution ===")
    dist = label_distribution(ds)
    for split_name, d in dist.items():
        print(f"  {split_name}: {d}")

    # 3. Token length analysis
    print("\n=== Token Length Analysis ===")
    plot_token_length_distribution(ds, results_dir)

    # 4. Generate plots
    print("\n=== Generating Plots ===")
    plot_context_length_distribution(ds, results_dir)
    print("  Saved: context_length_distribution.png")

    plot_holding_length_distribution(ds, results_dir)
    print("  Saved: holding_length_distribution.png")

    plot_label_distribution(ds, results_dir)
    print("  Saved: label_distribution.png")

    print(f"\nAll plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
