"""Phase 7: t-SNE visualization of SBERT embeddings colored by prediction correctness."""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from data_loader import load_casehold

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def get_embeddings(model, texts, batch_size=64):
    """Encode texts with SBERT."""
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def plot_tsne_by_label(embeddings, labels, title, filename):
    """t-SNE colored by true label (0-4)."""
    print(f"  Running t-SNE for {filename}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    labels_unique = sorted(set(labels))

    for label in labels_unique:
        mask = (labels == label)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[label], label=f"Holding {chr(65+label)}",
                   alpha=0.4, s=8, edgecolors="none")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, markerscale=3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_tsne_correct_vs_wrong(embeddings, correct_mask, title, filename):
    """t-SNE colored by correct vs incorrect prediction."""
    print(f"  Running t-SNE for {filename}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot incorrect first (background)
    wrong = ~correct_mask
    ax.scatter(coords[wrong, 0], coords[wrong, 1], c="#FF6B6B", label="Incorrect",
               alpha=0.3, s=8, edgecolors="none")
    # Plot correct on top
    ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1], c="#4CAF50", label="Correct",
               alpha=0.3, s=8, edgecolors="none")

    acc = np.mean(correct_mask) * 100
    ax.set_title(f"{title}\n(Accuracy: {acc:.1f}%)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, markerscale=3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_similarity_distribution(model, test_data, n_samples=1000):
    """Distribution of cosine similarities between context and correct/incorrect holdings."""
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.RandomState(42)
    indices = rng.choice(len(test_data), min(n_samples, len(test_data)), replace=False)
    samples = [test_data[i] for i in indices]

    correct_sims = []
    wrong_sims = []

    print("  Computing similarity distributions...")
    for ex in tqdm(samples, desc="Computing similarities"):
        ctx_emb = model.encode([ex["context"]])
        for i, holding in enumerate(ex["endings"]):
            hold_emb = model.encode([holding])
            sim = cosine_similarity(ctx_emb, hold_emb)[0, 0]
            if i == ex["label"]:
                correct_sims.append(sim)
            else:
                wrong_sims.append(sim)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(correct_sims, bins=50, alpha=0.6, color="#4CAF50", label="Correct Holding", density=True)
    ax.hist(wrong_sims, bins=50, alpha=0.6, color="#FF6B6B", label="Wrong Holdings", density=True)

    ax.axvline(np.mean(correct_sims), color="#2E7D32", linestyle="--", linewidth=2,
               label=f"Correct mean: {np.mean(correct_sims):.3f}")
    ax.axvline(np.mean(wrong_sims), color="#C62828", linestyle="--", linewidth=2,
               label=f"Wrong mean: {np.mean(wrong_sims):.3f}")

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Context-Holding Similarity Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "similarity_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: similarity_distribution.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples for t-SNE (default 2000 for speed)")
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = load_casehold()
    n = min(args.max_samples, len(ds["test"]))
    test_split = ds["test"].select(range(n))
    test_data = [test_split[i] for i in range(n)]
    labels = np.array([ex["label"] for ex in test_data])
    contexts = [ex["context"] for ex in test_data]

    print(f"Loading SBERT: {args.sbert_model}")
    sbert = SentenceTransformer(args.sbert_model)

    # Embed contexts
    print("Encoding test contexts...")
    embeddings = get_embeddings(sbert, contexts)

    # 1) t-SNE colored by true label
    plot_tsne_by_label(embeddings, labels, "t-SNE of Test Contexts (by True Label)", "tsne_by_label.png")

    # 2) t-SNE colored by SBERT correctness
    # Quick SBERT predictions
    print("Computing SBERT predictions for visualization...")
    sbert_preds = []
    for ex in tqdm(test_data, desc="SBERT predicting"):
        ctx_emb = sbert.encode([ex["context"]])
        hold_embs = sbert.encode(ex["endings"])
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(ctx_emb, hold_embs)[0]
        sbert_preds.append(np.argmax(sims))
    sbert_preds = np.array(sbert_preds)
    sbert_correct = (sbert_preds == labels)
    plot_tsne_correct_vs_wrong(embeddings, sbert_correct, "SBERT: Correct vs Incorrect", "tsne_sbert_correct.png")

    # 3) Load LoRA predictions if available
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            predictions = json.load(f)
        # prefer the large variant (the best model) when it is available
        key = ("LoRA FLAN-T5-large" if "LoRA FLAN-T5-large" in predictions
               else "LoRA FLAN-T5" if "LoRA FLAN-T5" in predictions else None)
        if key:
            lora_preds = np.array(predictions[key][:args.max_samples])
            lora_correct = (lora_preds == labels)
            plot_tsne_correct_vs_wrong(embeddings, lora_correct,
                                       f"{key}: Correct vs Incorrect", "tsne_lora_correct.png")

    # 4) Similarity distribution
    plot_similarity_distribution(sbert, test_data, n_samples=1000)

    print(f"\nAll embedding visualizations saved to: results/figures/")


if __name__ == "__main__":
    main()
