"""Phase 7: Error analysis — compare predictions across models, categorize failures."""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from peft import PeftModel
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from data_loader import load_casehold

LETTERS = ["A", "B", "C", "D", "E"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def get_bert_predictions(model_name, checkpoint_path, test_data, device, max_length=256):
    """Predictions + full 5-way logits from a fine-tuned encoder (logits kept for MRR)."""
    from baseline_bert import CaseHOLDDataset, MultipleChoiceModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = CaseHOLDDataset(test_data, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=8, num_workers=0)

    model = MultipleChoiceModel(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.eval()

    all_preds = []
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting ({model_name})"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_scores.append(logits.cpu().numpy())
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds), np.concatenate(all_scores, axis=0)


def get_lora_predictions(model_name, adapter_path, test_data, device):
    """Predictions + 5-way scores from a LoRA FLAN-T5.

    Scores each letter A-E by its first-decoder-step logit: same argmax as
    greedy decoding, but with a full ranking for MRR.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()

    letter_ids = [tokenizer(l, add_special_tokens=False).input_ids[0] for l in LETTERS]
    start_id = base_model.config.decoder_start_token_id

    all_preds = []
    all_scores = []
    for example in tqdm(test_data, desc="Predicting (LoRA)"):
        options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
        input_text = (
            f"Choose the correct holding for this legal case.\n\n"
            f"Context: {example['context']}\n\n"
            f"Options:\n{options}\n\nAnswer:"
        )
        inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(device)
        decoder_input_ids = torch.tensor([[start_id]], device=device)
        with torch.no_grad():
            output = model(**inputs, decoder_input_ids=decoder_input_ids)
        letter_logits = output.logits[0, 0, letter_ids].float().cpu().numpy()
        all_scores.append(letter_logits)
        all_preds.append(int(np.argmax(letter_logits)))
    return np.array(all_preds), np.stack(all_scores)


def get_sbert_predictions(test_data, model_name="all-MiniLM-L6-v2"):
    """Predictions + 5-way cosine similarities from the SBERT baseline."""
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(model_name)
    contexts = [ex["context"] for ex in test_data]
    ctx_emb = sbert.encode(contexts, batch_size=64, show_progress_bar=True)
    flat = [h for ex in test_data for h in ex["endings"]]
    hold_emb = sbert.encode(flat, batch_size=64, show_progress_bar=True)

    n = len(contexts)
    hold_emb = hold_emb.reshape(n, 5, -1)
    ctx_emb = ctx_emb / (np.linalg.norm(ctx_emb, axis=1, keepdims=True) + 1e-8)
    hold_emb = hold_emb / (np.linalg.norm(hold_emb, axis=2, keepdims=True) + 1e-8)
    scores = np.einsum("nd,nkd->nk", ctx_emb, hold_emb)
    return scores.argmax(axis=1), scores


def compute_mrr(scores, labels):
    """Mean Reciprocal Rank of the correct option (ties count as the worse rank)."""
    n = len(labels)
    correct_scores = scores[np.arange(n), labels]
    ranks = (scores >= correct_scores[:, None]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def analyze_context_length_errors(test_data, labels, preds, model_name):
    """Analyze error rate by context length."""
    lengths = [len(ex["context"].split()) for ex in test_data]
    correct = (preds == labels)

    # Bin by context length
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
    bin_labels = []
    bin_accs = []

    for lo, hi in bins:
        mask = [(lo <= l < hi) for l in lengths]
        if sum(mask) > 0:
            bin_correct = [c for c, m in zip(correct, mask) if m]
            bin_labels.append(f"{lo}-{hi}")
            bin_accs.append(np.mean(bin_correct) * 100)

    return bin_labels, bin_accs


def analyze_holding_length_errors(test_data, labels, preds):
    """Analyze error rate by correct holding length."""
    holding_lengths = [len(ex["endings"][ex["label"]].split()) for ex in test_data]
    correct = (preds == labels)

    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]
    bin_labels = []
    bin_accs = []

    for lo, hi in bins:
        mask = [(lo <= l < hi) for l in holding_lengths]
        if sum(mask) > 0:
            bin_correct = [c for c, m in zip(correct, mask) if m]
            bin_labels.append(f"{lo}-{hi}")
            bin_accs.append(np.mean(bin_correct) * 100)

    return bin_labels, bin_accs


def plot_confusion_matrix(labels, preds, model_name):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(5):
        for j in range(5):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(LETTERS)
    ax.set_yticklabels(LETTERS)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Proportion")

    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f"confusion_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_error_by_context_length(all_results, test_data, labels):
    """Compare error rates by context length across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"SBERT": "#7FB3D8", "BERT": "#90EE90", "RoBERTa": "#66BB66",
              "LoRA FLAN-T5": "#FF6B6B", "LoRA FLAN-T5-large": "#CC2222",
              "Legal-BERT (custom)": "#3388CC", "Legal-BERT (nlpaueb)": "#225588"}
    for name, preds in all_results.items():
        bin_labels, bin_accs = analyze_context_length_errors(test_data, labels, preds, name)
        ax.plot(bin_labels, bin_accs, "o-", label=name, color=colors.get(name, "gray"), linewidth=2, markersize=8)

    ax.set_xlabel("Context Length (words)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Context Length", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "error_by_context_length.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_error_by_holding_length(all_results, test_data, labels):
    """Compare error rates by holding length across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"SBERT": "#7FB3D8", "BERT": "#90EE90", "RoBERTa": "#66BB66",
              "LoRA FLAN-T5": "#FF6B6B", "LoRA FLAN-T5-large": "#CC2222",
              "Legal-BERT (custom)": "#3388CC", "Legal-BERT (nlpaueb)": "#225588"}
    for name, preds in all_results.items():
        bin_labels, bin_accs = analyze_holding_length_errors(test_data, labels, preds)
        ax.plot(bin_labels, bin_accs, "o-", label=name, color=colors.get(name, "gray"), linewidth=2, markersize=8)

    ax.set_xlabel("Correct Holding Length (words)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Holding Length", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "error_by_holding_length.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_agreement_venn(all_results, labels):
    """Analyze model agreement: what each model gets right/wrong uniquely."""
    model_names = list(all_results.keys())
    correct_sets = {}
    for name, preds in all_results.items():
        correct_sets[name] = set(np.where(preds == labels)[0])

    # Pairwise agreement
    n = len(labels)
    print("\n  Model Agreement Analysis:")
    print(f"  {'':>20}", end="")
    for name in model_names:
        print(f"  {name:>15}", end="")
    print()

    for name1 in model_names:
        print(f"  {name1:>20}", end="")
        for name2 in model_names:
            overlap = len(correct_sets[name1] & correct_sets[name2])
            print(f"  {overlap/n*100:>14.1f}%", end="")
        print()

    # Unique correct predictions per model
    print("\n  Unique correct predictions (only this model got right):")
    all_correct = set.union(*correct_sets.values()) if correct_sets else set()
    for name in model_names:
        others = set.union(*[s for n, s in correct_sets.items() if n != name]) if len(model_names) > 1 else set()
        unique = correct_sets[name] - others
        print(f"    {name}: {len(unique)} samples ({len(unique)/n*100:.1f}%)")

    # All models correct / all wrong
    all_right = set.intersection(*correct_sets.values()) if correct_sets else set()
    all_wrong = set(range(n)) - all_correct
    print(f"\n  All models correct: {len(all_right)} ({len(all_right)/n*100:.1f}%)")
    print(f"  All models wrong:   {len(all_wrong)} ({len(all_wrong)/n*100:.1f}%)")

    return correct_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick test")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading dataset...")
    ds = load_casehold()
    test_data = ds["test"]
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    labels = np.array([ex["label"] for ex in test_data])

    all_results = {}
    all_scores = {}

    # SBERT predictions (cosine similarity ranking, no fine-tuning)
    print("\nGetting SBERT predictions...")
    all_results["SBERT"], all_scores["SBERT"] = get_sbert_predictions(test_data)
    plot_confusion_matrix(labels, all_results["SBERT"], "SBERT")

    # BERT predictions
    bert_ckpt = os.path.join(RESULTS_DIR, "bert-base-uncased_best.pt")
    if os.path.exists(bert_ckpt):
        print("\nGetting BERT predictions...")
        all_results["BERT"], all_scores["BERT"] = get_bert_predictions(
            "bert-base-uncased", bert_ckpt, test_data, device)
        plot_confusion_matrix(labels, all_results["BERT"], "BERT")

    # RoBERTa predictions
    roberta_ckpt = os.path.join(RESULTS_DIR, "roberta-base_best.pt")
    if os.path.exists(roberta_ckpt):
        print("\nGetting RoBERTa predictions...")
        all_results["RoBERTa"], all_scores["RoBERTa"] = get_bert_predictions(
            "roberta-base", roberta_ckpt, test_data, device)
        plot_confusion_matrix(labels, all_results["RoBERTa"], "RoBERTa")

    # Legal-BERT (CaseHOLD custom) predictions
    legalbert_ckpt = os.path.join(RESULTS_DIR, "casehold_custom-legalbert_best.pt")
    if os.path.exists(legalbert_ckpt):
        print("\nGetting Legal-BERT (custom) predictions...")
        all_results["Legal-BERT (custom)"], all_scores["Legal-BERT (custom)"] = get_bert_predictions(
            "casehold/custom-legalbert", legalbert_ckpt, test_data, device)
        plot_confusion_matrix(labels, all_results["Legal-BERT (custom)"], "Legal-BERT (custom)")

    # Legal-BERT (NLPAUEB) predictions
    nlpaueb_ckpt = os.path.join(RESULTS_DIR, "nlpaueb_legal-bert-base-uncased_best.pt")
    if os.path.exists(nlpaueb_ckpt):
        print("\nGetting Legal-BERT (nlpaueb) predictions...")
        all_results["Legal-BERT (nlpaueb)"], all_scores["Legal-BERT (nlpaueb)"] = get_bert_predictions(
            "nlpaueb/legal-bert-base-uncased", nlpaueb_ckpt, test_data, device)
        plot_confusion_matrix(labels, all_results["Legal-BERT (nlpaueb)"], "Legal-BERT (nlpaueb)")

    # LoRA predictions
    lora_path = os.path.join(RESULTS_DIR, "google_flan-t5-base_lora_best")
    if os.path.exists(lora_path):
        print("\nGetting LoRA FLAN-T5 predictions...")
        all_results["LoRA FLAN-T5"], all_scores["LoRA FLAN-T5"] = get_lora_predictions(
            "google/flan-t5-base", lora_path, test_data, device)
        plot_confusion_matrix(labels, all_results["LoRA FLAN-T5"], "LoRA FLAN-T5")

    # LoRA FLAN-T5-large predictions
    lora_large_path = os.path.join(RESULTS_DIR, "google_flan-t5-large_lora_best")
    if os.path.exists(lora_large_path):
        print("\nGetting LoRA FLAN-T5-large predictions...")
        all_results["LoRA FLAN-T5-large"], all_scores["LoRA FLAN-T5-large"] = get_lora_predictions(
            "google/flan-t5-large", lora_large_path, test_data, device)
        plot_confusion_matrix(labels, all_results["LoRA FLAN-T5-large"], "LoRA FLAN-T5-large")

    if not all_results:
        print("No model checkpoints found. Skipping error analysis.")
        return

    # Error analysis plots
    print("\nGenerating error analysis plots...")
    plot_error_by_context_length(all_results, test_data, labels)
    plot_error_by_holding_length(all_results, test_data, labels)

    # Agreement analysis
    plot_agreement_venn(all_results, labels)

    # MRR from the full 5-way scores (Gemini excluded: API runs only kept
    # the final answer letter per sample)
    print("\n  Mean Reciprocal Rank (MRR):")
    mrr_results = {
        "note": ("Gemini 2.5 Flash/Pro are excluded: the API experiments stored only "
                 "the final predicted letter per sample, so no ranking over the five "
                 "candidates exists; recomputing would require re-running paid API calls."),
        "random_baseline": round(sum(1.0 / k for k in range(1, 6)) / 5, 4),
    }
    for name, scores in all_scores.items():
        mrr = compute_mrr(scores, labels)
        mrr_results[name] = round(mrr, 4)
        print(f"    {name}: {mrr:.4f}")
    print(f"    (random baseline: {mrr_results['random_baseline']:.4f})")
    mrr_path = os.path.join(RESULTS_DIR, "mrr.json")
    with open(mrr_path, "w") as f:
        json.dump(mrr_results, f, indent=2)
    print(f"  MRR saved to: {os.path.basename(mrr_path)}")

    # Save predictions for further analysis
    predictions = {name: preds.tolist() for name, preds in all_results.items()}
    predictions["labels"] = labels.tolist()
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    print(f"\n  Predictions saved to: {os.path.basename(pred_path)}")

    # Print per-model classification reports
    for name, preds in all_results.items():
        print(f"\n{'='*60}")
        print(f"Classification Report: {name}")
        print(f"{'='*60}")
        print(classification_report(labels, preds, target_names=LETTERS, digits=4))

    print(f"\nAll figures saved to: results/figures/")


if __name__ == "__main__":
    main()
