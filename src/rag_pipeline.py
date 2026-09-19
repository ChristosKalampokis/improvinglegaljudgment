"""RAG-Legal: Retrieval-Augmented Generation for CaseHOLD."""

import os
import json
import argparse
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_loader import load_casehold


LETTERS = ["A", "B", "C", "D", "E"]


def build_retrieval_index(train_data, sbert_model, batch_size=64, include_holdings=False):
    """Build a FAISS index over training contexts.

    include_holdings indexes context + correct holding instead; queries stay
    context-only (the correct holding is unknown at test time).
    """
    print("Building retrieval index...")
    if include_holdings:
        contexts = [f"{ctx} {endings[label]}"
                    for ctx, endings, label in zip(train_data["context"],
                                                   train_data["endings"],
                                                   train_data["label"])]
    else:
        contexts = train_data["context"]
    embeddings = sbert_model.encode(contexts, batch_size=batch_size, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    print(f"  Index built with {index.ntotal} vectors (dim={dim})")
    return index, embeddings


def retrieve_similar(query_text, sbert_model, index, train_data, k=3):
    """Retrieve top-k similar training examples for a query context."""
    query_emb = sbert_model.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, k)
    retrieved = []
    for idx in indices[0]:
        retrieved.append(train_data[int(idx)])
    return retrieved


def build_rag_prompt(example, retrieved_examples):
    """Build a prompt with retrieved examples as context."""
    prompt = "Choose the correct legal holding for the case. Here are similar cases for reference:\n\n"

    # Adjust truncation based on number of retrieved examples
    ctx_limit = max(100, 300 // len(retrieved_examples)) if retrieved_examples else 200
    hold_limit = max(80, 200 // len(retrieved_examples)) if retrieved_examples else 100

    for i, ret in enumerate(retrieved_examples):
        prompt += (
            f"Similar case {i+1}:\n"
            f"Context: {ret['context'][:ctx_limit]}...\n"
            f"Correct holding: {LETTERS[ret['label']]}) {ret['endings'][ret['label']][:hold_limit]}...\n\n"
        )

    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    prompt += (
        f"---\n\n"
        f"Now choose the correct holding for this case:\n\n"
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )
    return prompt


def extract_answer(generated_text):
    """Extract the answer letter (A-E) from model output."""
    text = generated_text.strip().upper()
    for letter in LETTERS:
        if text.startswith(letter):
            return letter
    for letter in LETTERS:
        if letter in text:
            return letter
    return "A"


def run_rag_experiment(model, tokenizer, sbert_model, index, train_data, test_data,
                       k=3, device="cuda", max_samples=None, max_input_length=1024):
    """Run RAG experiment: retrieve + generate for each test sample."""
    preds = []
    labels = []

    samples = list(test_data)
    if max_samples:
        samples = samples[:max_samples]

    for example in tqdm(samples, desc=f"RAG (k={k})"):
        # Retrieve similar cases
        retrieved = retrieve_similar(example["context"], sbert_model, index, train_data, k=k)

        # Build RAG prompt
        prompt = build_rag_prompt(example, retrieved)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_letter = extract_answer(generated)
        pred_idx = LETTERS.index(pred_letter) if pred_letter in LETTERS else 0

        preds.append(pred_idx)
        labels.append(example["label"])

    acc = accuracy_score(labels, preds)
    return acc, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model", type=str, default="google/flan-t5-base")
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--k_values", type=str, default="0,1,3,5",
                        help="Comma-separated k values for ablation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lora_adapter", type=str, default=None,
                        help="LoRA adapter to load on top of the LLM (RAG + LoRA hybrid)")
    parser.add_argument("--index_holdings", action="store_true",
                        help="Index context + correct holding instead of context only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading dataset...")
    ds = load_casehold()

    # Plain BERT models get auto-wrapped with mean pooling by the library
    print(f"Loading retriever: {args.sbert_model}")
    sbert_model = SentenceTransformer(args.sbert_model)
    if sbert_model.max_seq_length is None or sbert_model.max_seq_length > 256:
        sbert_model.max_seq_length = 256

    # Build index
    index, _ = build_retrieval_index(ds["train"], sbert_model,
                                     include_holdings=args.index_holdings)

    # Load LLM
    print(f"Loading LLM: {args.llm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.llm_model).to(device)

    # RAG + LoRA hybrid; with k=0 this doubles as the no-retrieval baseline
    if args.lora_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.lora_adapter}")
        model = PeftModel.from_pretrained(model, args.lora_adapter).to(device)
    model.eval()

    eval_split = ds[args.split]
    k_values = [int(k) for k in args.k_values.split(",")]
    results = {"llm_model": args.llm_model, "sbert_model": args.sbert_model, "split": args.split,
               "lora_adapter": args.lora_adapter, "index_holdings": args.index_holdings}

    for k in k_values:
        print(f"\n=== RAG with k={k} ===")
        if k == 0:
            # No retrieval — plain prompting (baseline for ablation)
            from prompt_engineering import build_zero_shot_prompt, run_experiment
            acc, _ = run_experiment(
                model, tokenizer, eval_split, build_zero_shot_prompt, device,
                max_samples=args.max_samples,
            )
        else:
            acc, _ = run_rag_experiment(
                model, tokenizer, sbert_model, index, ds["train"], eval_split,
                k=k, device=device, max_samples=args.max_samples,
            )
        print(f"  k={k} accuracy: {acc:.4f}")
        results[f"k{k}_accuracy"] = acc

    # Non-default options get filename tags so ablation runs don't overwrite
    # each other; the default config keeps the original filename.
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_name = args.llm_model.replace("/", "_")
    tags = ""
    if args.sbert_model != "all-MiniLM-L6-v2":
        tags += f"_retr_{args.sbert_model.replace('/', '_')}"
    if args.index_holdings:
        tags += "_holdix"
    if args.lora_adapter:
        tags += "_lora"
    results_path = os.path.join(results_dir, f"rag_{save_name}{tags}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {os.path.basename(results_path)}")


if __name__ == "__main__":
    main()
