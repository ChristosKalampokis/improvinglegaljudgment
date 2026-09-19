"""Semantic score: cosine similarity between predicted and true holdings.

Correct predictions count as 1.0; the errors-only mean shows how close the
misses are. Needs results/predictions.json from error_analysis.py.
"""

import os
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from data_loader import load_casehold

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    if not os.path.exists(pred_path):
        print("results/predictions.json not found - run error_analysis.py first.")
        return

    with open(pred_path) as f:
        predictions = json.load(f)
    labels = np.array(predictions.pop("labels"))
    n = len(labels)

    print("Loading dataset...")
    ds = load_casehold()
    all_endings = ds["test"]["endings"][:n]

    print(f"Loading SBERT: {args.sbert_model}")
    sbert = SentenceTransformer(args.sbert_model)

    # Encode all 5 candidate holdings per test sample once
    flat = [h for endings in all_endings for h in endings]
    print(f"Encoding {len(flat)} candidate holdings...")
    embs = sbert.encode(flat, batch_size=64, show_progress_bar=True)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    embs = embs.reshape(n, 5, -1)

    true_emb = embs[np.arange(n), labels]

    results = {
        "sbert_model": args.sbert_model,
        "n_samples": n,
        "note": ("Gemini 2.5 Flash/Pro are excluded: the API experiments stored only "
                 "aggregate accuracies, not per-sample predictions, so the predicted "
                 "holding text is unknown; recomputing would require re-running paid "
                 "API calls."),
    }
    print(f"\n{'Model':<25} {'Accuracy':>10} {'SemScore':>10} {'SemScore(err)':>14}")
    print("-" * 62)

    for name, preds in predictions.items():
        preds = np.array(preds[:n])
        pred_emb = embs[np.arange(n), preds]
        sims = np.sum(pred_emb * true_emb, axis=1)

        correct = (preds == labels)
        score_all = float(np.mean(sims))
        score_err = float(np.mean(sims[~correct])) if (~correct).any() else 1.0
        acc = float(np.mean(correct))

        results[name] = {
            "accuracy": round(acc, 4),
            "semantic_score_all": round(score_all, 4),
            "semantic_score_errors_only": round(score_err, 4),
        }
        print(f"{name:<25} {acc:>10.4f} {score_all:>10.4f} {score_err:>14.4f}")

    out_path = os.path.join(RESULTS_DIR, "semantic_scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
