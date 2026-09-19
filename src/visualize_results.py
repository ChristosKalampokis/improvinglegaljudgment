"""Phase 7: Generate publication-quality visualizations of all experiment results."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_summary():
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        return json.load(f)


def plot_main_comparison(summary):
    """Bar chart comparing all methods side by side."""
    methods = [
        ("Random", 0.20, "#CCCCCC"),
        ("SBERT\n(zero-shot)", summary["sbert"]["test_accuracy"], "#7FB3D8"),
        ("FLAN-T5\nzero-shot", summary["prompt_google_flan-t5-base"]["zero_shot_accuracy"], "#FFA07A"),
        ("FLAN-T5\nfew-shot", summary["prompt_google_flan-t5-base"]["few_shot_accuracy"], "#FFA07A"),
        ("FLAN-T5\nCoT", summary["prompt_google_flan-t5-base"]["cot_accuracy"], "#FFA07A"),
        ("RAG k=3", summary["rag_google_flan-t5-base"]["k3_accuracy"], "#FFD700"),
        ("BERT\n(fine-tuned)", summary["bert-base-uncased"]["test_accuracy"], "#90EE90"),
        ("RoBERTa\n(fine-tuned)", summary["roberta-base"]["test_accuracy"], "#90EE90"),
    ]

    # Legal-BERT variants (added conditionally once trained)
    if "casehold_custom-legalbert" in summary:
        methods.append(("Legal-BERT\n(custom)", summary["casehold_custom-legalbert"]["test_accuracy"], "#3388CC"))
    if "nlpaueb_legal-bert-base-uncased" in summary:
        methods.append(("Legal-BERT\n(nlpaueb)", summary["nlpaueb_legal-bert-base-uncased"]["test_accuracy"], "#225588"))

    methods.append(("LoRA FLAN-T5\n(ours)", summary["lora_google_flan-t5-base"]["test_accuracy"], "#FF6B6B"))
    if "lora_google_flan-t5-large" in summary:
        methods.append(("LoRA FLAN-T5\nlarge (ours)", summary["lora_google_flan-t5-large"]["test_accuracy"], "#CC2222"))

    # Gemini cloud models (added conditionally)
    if "gemini_gemini_2.5_flash" in summary:
        methods.append(("Gemini Flash\n(RAG k=3)", summary["gemini_gemini_2.5_flash"]["rag_k3_accuracy"], "#B39DDB"))
    if "gemini_gemini_2.5_pro" in summary:
        methods.append(("Gemini Pro\n(RAG k=5)", summary["gemini_gemini_2.5_pro"]["rag_k5_accuracy"], "#7E57C2"))

    names = [m[0] for m in methods]
    accs = [m[1] for m in methods]
    colors = [m[2] for m in methods]

    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(range(len(names)), [a * 100 for a in accs], color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("CaseHOLD: Comparison of All Methods", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="Random chance (20%)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color="#CCCCCC", label="Random Baseline"),
        mpatches.Patch(color="#7FB3D8", label="Embedding Baseline"),
        mpatches.Patch(color="#FFA07A", label="Prompt Engineering"),
        mpatches.Patch(color="#FFD700", label="RAG Pipeline"),
        mpatches.Patch(color="#90EE90", label="Supervised Fine-tuning"),
        mpatches.Patch(color="#3388CC", label="Legal-Domain Pretraining"),
        mpatches.Patch(color="#FF6B6B", label="LoRA Fine-tuning (Ours)"),
        mpatches.Patch(color="#7E57C2", label="Gemini (Cloud LLM)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "main_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_method_groups(summary):
    """Grouped bar chart: baselines vs prompting vs RAG vs fine-tuned."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    # Group 1: Baselines
    ax = axes[0]
    names = ["Random", "SBERT"]
    accs = [20.0, summary["sbert"]["test_accuracy"] * 100]
    ax.bar(names, accs, color=["#CCCCCC", "#7FB3D8"], edgecolor="black", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 1, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Baselines", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(axis="y", alpha=0.3)

    # Group 2: Prompt Engineering
    ax = axes[1]
    names = ["Zero-shot", "Few-shot", "CoT"]
    accs = [
        summary["prompt_google_flan-t5-base"]["zero_shot_accuracy"] * 100,
        summary["prompt_google_flan-t5-base"]["few_shot_accuracy"] * 100,
        summary["prompt_google_flan-t5-base"]["cot_accuracy"] * 100,
    ]
    ax.bar(names, accs, color="#FFA07A", edgecolor="black", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 1, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Prompt Engineering\n(FLAN-T5-base)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Group 3: RAG
    ax = axes[2]
    names = ["k=0", "k=1", "k=3", "k=5"]
    accs = [
        summary["rag_google_flan-t5-base"]["k0_accuracy"] * 100,
        summary["rag_google_flan-t5-base"]["k1_accuracy"] * 100,
        summary["rag_google_flan-t5-base"]["k3_accuracy"] * 100,
        summary["rag_google_flan-t5-base"]["k5_accuracy"] * 100,
    ]
    ax.bar(names, accs, color="#FFD700", edgecolor="black", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 1, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("RAG Pipeline\n(FAISS + FLAN-T5)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Group 4: Fine-tuned
    ax = axes[3]
    names = ["BERT", "RoBERTa"]
    accs = [
        summary["bert-base-uncased"]["test_accuracy"] * 100,
        summary["roberta-base"]["test_accuracy"] * 100,
    ]
    colors = ["#90EE90", "#90EE90"]
    if "casehold_custom-legalbert" in summary:
        names.append("Legal-BERT\n(custom)")
        accs.append(summary["casehold_custom-legalbert"]["test_accuracy"] * 100)
        colors.append("#3388CC")
    if "nlpaueb_legal-bert-base-uncased" in summary:
        names.append("Legal-BERT\n(nlpaueb)")
        accs.append(summary["nlpaueb_legal-bert-base-uncased"]["test_accuracy"] * 100)
        colors.append("#225588")
    names.append("LoRA\nFLAN-T5")
    accs.append(summary["lora_google_flan-t5-base"]["test_accuracy"] * 100)
    colors.append("#FF6B6B")
    if "lora_google_flan-t5-large" in summary:
        names.append("LoRA\nFLAN-T5-large")
        accs.append(summary["lora_google_flan-t5-large"]["test_accuracy"] * 100)
        colors.append("#CC2222")
    ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 1, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Fine-tuned Models", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("CaseHOLD Results by Method Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "method_groups.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_rag_ablation(summary):
    """Line plot showing RAG accuracy vs number of retrieved examples."""
    k_values = [0, 1, 3, 5]
    accs = [
        summary["rag_google_flan-t5-base"]["k0_accuracy"] * 100,
        summary["rag_google_flan-t5-base"]["k1_accuracy"] * 100,
        summary["rag_google_flan-t5-base"]["k3_accuracy"] * 100,
        summary["rag_google_flan-t5-base"]["k5_accuracy"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, accs, "o-", color="#FFD700", linewidth=2, markersize=10, markeredgecolor="black")

    for k, a in zip(k_values, accs):
        ax.annotate(f"{a:.2f}%", (k, a), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold")

    # Reference lines
    ax.axhline(y=summary["sbert"]["test_accuracy"] * 100, color="#7FB3D8",
               linestyle="--", alpha=0.7, label=f"SBERT baseline ({summary['sbert']['test_accuracy']*100:.1f}%)")
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="Random (20%)")

    ax.set_xlabel("Number of Retrieved Examples (k)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("RAG Ablation: Effect of Retrieved Examples on Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(k_values)
    ax.set_ylim(15, 60)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rag_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_retriever_ablation(summary):
    """Compare RAG retrievers: MiniLM vs Legal-BERT (mean pooling) vs BGE.

    Skips silently unless at least two retriever result files exist.
    """
    retrievers = [
        ("MiniLM (all-MiniLM-L6-v2)", "rag_google_flan-t5-base", "#FFD700"),
        ("Legal-BERT (mean pooling)", "rag_google_flan-t5-base_retr_nlpaueb_legal-bert-base-uncased", "#3388CC"),
        ("BGE (bge-base-en-v1.5)", "rag_google_flan-t5-base_retr_BAAI_bge-base-en-v1.5", "#8E44AD"),
    ]
    available = [(label, summary[key], color) for label, key, color in retrievers if key in summary]
    if len(available) < 2:
        return

    k_values = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, res, color in available:
        ks = [k for k in k_values if f"k{k}_accuracy" in res]
        accs = [res[f"k{k}_accuracy"] * 100 for k in ks]
        ax.plot(ks, accs, "o-", label=label, color=color, linewidth=2, markersize=9,
                markeredgecolor="black")
        for k, a in zip(ks, accs):
            ax.annotate(f"{a:.1f}%", (k, a), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, fontweight="bold")

    # No-retrieval reference (k=0 from the original MiniLM run)
    base = summary.get("rag_google_flan-t5-base", {}).get("k0_accuracy")
    if base is not None:
        ax.axhline(y=base * 100, color="gray", linestyle="--", alpha=0.6,
                   label=f"No retrieval, k=0 ({base*100:.1f}%)")

    ax.set_xlabel("Number of Retrieved Examples (k)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("RAG Retriever Ablation (generator: FLAN-T5-base)", fontsize=13, fontweight="bold")
    ax.set_xticks(k_values)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "retriever_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_lora_epoch_comparison(summary):
    """Compare 1-epoch vs 3-epoch LoRA (if both results available)."""
    # We know from the experiment: 1-epoch = 80.75%, 3-epoch = 85.22%
    epochs = [1, 3]
    accs = [80.75, summary["lora_google_flan-t5-base"]["test_accuracy"] * 100]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["1 Epoch", "3 Epochs"], accs, color=["#FFAAAA", "#FF6B6B"],
                  edgecolor="black", linewidth=0.5, width=0.5)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{a:.2f}%", ha="center", fontsize=12, fontweight="bold")

    # Reference lines
    ax.axhline(y=summary["roberta-base"]["test_accuracy"] * 100, color="#90EE90",
               linestyle="--", alpha=0.7, label=f"RoBERTa ({summary['roberta-base']['test_accuracy']*100:.1f}%)")
    ax.axhline(y=summary["bert-base-uncased"]["test_accuracy"] * 100, color="#66BB66",
               linestyle="--", alpha=0.7, label=f"BERT ({summary['bert-base-uncased']['test_accuracy']*100:.1f}%)")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("LoRA Fine-tuning: Effect of Training Epochs", fontsize=13, fontweight="bold")
    ax.set_ylim(65, 90)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "lora_epochs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_improvement_waterfall(summary):
    """Waterfall chart showing incremental improvements from baseline to best."""
    # Best supervised encoder (BERT / RoBERTa / Legal-BERT variants)
    supervised_keys = ["bert-base-uncased", "roberta-base",
                       "casehold_custom-legalbert", "nlpaueb_legal-bert-base-uncased"]
    best_supervised = max(summary[k]["test_accuracy"] for k in supervised_keys if k in summary)

    stages = [
        ("Random\nBaseline", 20.0),
        ("+ SBERT\nEmbeddings", summary["sbert"]["test_accuracy"] * 100),
        ("+ Prompt\nEngineering", summary["prompt_google_flan-t5-base"]["few_shot_accuracy"] * 100),
        ("+ RAG\nRetrieval", summary["rag_google_flan-t5-base"]["k3_accuracy"] * 100),
        ("+ Supervised\nFine-tuning\n(Legal-BERT)", best_supervised * 100),
        ("+ LoRA\n(Ours)", summary["lora_google_flan-t5-base"]["test_accuracy"] * 100),
    ]
    if "lora_google_flan-t5-large" in summary:
        stages.append(("+ Scale\n(LoRA large)",
                       summary["lora_google_flan-t5-large"]["test_accuracy"] * 100))

    names = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw bars showing each level
    bar_colors = ["#CCCCCC", "#7FB3D8", "#FFA07A", "#FFD700", "#90EE90", "#FF6B6B", "#CC2222"]
    bars = ax.bar(range(len(names)), values, color=bar_colors[:len(names)],
                  edgecolor="black", linewidth=0.5)

    # Add value and delta labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
        if i > 0:
            # place the delta inside the bar: the midpoint between consecutive
            # values collides with the value label when two bars are similar
            delta = values[i] - values[i-1]
            color = "darkgreen" if delta > 0 else "darkred"
            ax.annotate(f"{delta:+.1f}%", (i, values[i] * 0.55),
                        ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Progressive Improvement: From Baseline to Best Pipeline", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "improvement_waterfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary = load_summary()

    print("Generating visualizations...")
    plot_main_comparison(summary)
    plot_method_groups(summary)
    plot_rag_ablation(summary)
    plot_retriever_ablation(summary)
    plot_lora_epoch_comparison(summary)
    plot_improvement_waterfall(summary)
    print(f"\nAll figures saved to: results/figures/")


if __name__ == "__main__":
    main()
