"""SBERT embedding baseline for CaseHOLD — zero-shot cosine similarity ranking."""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_loader import load_casehold


def evaluate_sbert(model, split, batch_size=64):
    """Rank candidate holdings by cosine similarity with the context."""
    contexts = split["context"]
    all_endings = split["endings"]
    labels = split["label"]

    # Encode all contexts
    print("  Encoding contexts...")
    context_embeddings = model.encode(contexts, batch_size=batch_size, show_progress_bar=True)

    # Encode all holdings (5 per sample, flattened)
    flat_holdings = [h for endings in all_endings for h in endings]
    print("  Encoding holdings...")
    holding_embeddings = model.encode(flat_holdings, batch_size=batch_size, show_progress_bar=True)
    holding_embeddings = holding_embeddings.reshape(len(contexts), 5, -1)

    # Cosine similarity: pick the holding closest to the context
    preds = []
    for i in range(len(contexts)):
        ctx_emb = context_embeddings[i]
        h_embs = holding_embeddings[i]
        # Cosine similarity
        sims = np.dot(h_embs, ctx_emb) / (
            np.linalg.norm(h_embs, axis=1) * np.linalg.norm(ctx_emb) + 1e-8
        )
        preds.append(int(np.argmax(sims)))

    acc = accuracy_score(labels, preds)
    return acc, preds


def main():
    print("Loading dataset...")
    ds = load_casehold()

    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    results = {"model": f"sbert-{model_name}"}

    for split_name in ["validation", "test"]:
        print(f"\nEvaluating on {split_name}...")
        acc, _ = evaluate_sbert(model, ds[split_name])
        print(f"  {split_name} accuracy: {acc:.4f}")
        results[f"{split_name}_accuracy"] = acc

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "sbert_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
