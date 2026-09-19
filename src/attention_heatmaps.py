"""Attention heatmaps: which tokens does [CLS] attend to on a (context, holding) pair?

Picks test cases where Legal-BERT is right and BERT is wrong (via
predictions.json) and plots each model's top attended tokens. Local models
only - APIs like Gemini don't expose attention weights.
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from data_loader import load_casehold
from baseline_bert import MultipleChoiceModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# (display name, HF model id, checkpoint filename, predictions.json key)
# custom-legalbert is skipped here: its vocab has no punctuation tokens, so
# ~25% of tokens render as [UNK] and the figures become unreadable.
MODELS = [
    ("BERT", "bert-base-uncased", "bert-base-uncased_best.pt", "BERT"),
    ("Legal-BERT (nlpaueb)", "nlpaueb/legal-bert-base-uncased",
     "nlpaueb_legal-bert-base-uncased_best.pt", "Legal-BERT (nlpaueb)"),
]

SKIP_TOKENS = {"[PAD]", "[CLS]"}


def cls_attention(model, tokenizer, context, holding, device, max_length=256):
    """Return (tokens, per-token [CLS] attention) for a (context, holding) pair."""
    enc = tokenizer(context, holding, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.encoder(**enc, output_attentions=True)
    # Average over all layers and heads; the last layer alone is near-uniform
    att = torch.stack(out.attentions)          # (layers, 1, heads, seq, seq)
    cls_att = att[:, 0].mean(dim=(0, 1))[0]    # CLS row: (seq,)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, cls_att.cpu().numpy()


def plot_example(example_idx, example, model_data, out_path):
    """One figure per test example: a row per model with its top attended tokens."""
    n_models = len(model_data)
    fig, axes = plt.subplots(n_models, 1, figsize=(11, 4.2 * n_models))
    if n_models == 1:
        axes = [axes]

    for ax, (display_name, tokens, att, correct) in zip(axes, model_data):
        # Rank real tokens by attention weight
        pairs = [(t, a) for t, a in zip(tokens, att) if t not in SKIP_TOKENS]
        pairs.sort(key=lambda x: -x[1])
        top = pairs[:15][::-1]  # reversed for horizontal bar order

        names = [t for t, _ in top]
        weights = [a for _, a in top]
        colors = ["#3388CC" if n not in ("[SEP]",) else "#AAAAAA" for n in names]

        ax.barh(range(len(top)), weights, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=9, fontfamily="monospace")
        ax.set_xlabel("Mean [CLS] attention (averaged over all layers and heads)", fontsize=10)
        marker = "correct" if correct else "WRONG"
        ax.set_title(f"{display_name} — prediction {marker}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Top attended tokens — test example #{example_idx}\n"
                 f"(context, correct holding) pair", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset...")
    ds = load_casehold()
    test_data = ds["test"]

    # Keep only models whose checkpoints exist
    available = [(n, mid, ckpt, key) for n, mid, ckpt, key in MODELS
                 if os.path.exists(os.path.join(RESULTS_DIR, ckpt))]
    if not available:
        print("No fine-tuned checkpoints found — run baseline_bert.py first.")
        return

    # Pick illustrative examples: Legal-BERT right, BERT wrong (if predictions exist)
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    chosen = None
    predictions = {}
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            predictions = json.load(f)
        labels = np.array(predictions.get("labels", []))
        if "Legal-BERT (nlpaueb)" in predictions and "BERT" in predictions and len(labels):
            lb = np.array(predictions["Legal-BERT (nlpaueb)"])
            bb = np.array(predictions["BERT"])
            interesting = np.where((lb == labels) & (bb != labels))[0]
            if len(interesting) >= args.num_examples:
                chosen = interesting[:args.num_examples].tolist()
                print(f"Selected {len(chosen)} examples where Legal-BERT is correct and BERT is not.")
    if chosen is None:
        chosen = list(range(args.num_examples))
        print(f"Using the first {args.num_examples} test examples.")

    # Load each model once, compute attention for all chosen examples
    per_example = {i: [] for i in chosen}
    for display_name, model_id, ckpt, pred_key in available:
        print(f"\nLoading {display_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # eager attention required: the default SDPA kernel does not return
        # attention weights (out.attentions would be empty)
        model = MultipleChoiceModel(model_id, attn_implementation="eager").to(device)
        model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, ckpt),
                                         weights_only=True, map_location=device))
        model.eval()

        model_preds = predictions.get(pred_key)
        for i in chosen:
            ex = test_data[int(i)]
            tokens, att = cls_attention(model, tokenizer, ex["context"],
                                        ex["endings"][ex["label"]], device,
                                        max_length=args.max_length)
            correct = (model_preds is not None and model_preds[int(i)] == ex["label"])
            per_example[i].append((display_name, tokens, att, correct))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nGenerating attention figures...")
    for i in chosen:
        out_path = os.path.join(FIGURES_DIR, f"attention_example_{i}.png")
        plot_example(i, test_data[int(i)], per_example[i], out_path)

    print(f"\nAll attention heatmaps saved to: results/figures/")


if __name__ == "__main__":
    main()
