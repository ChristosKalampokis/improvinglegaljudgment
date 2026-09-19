"""Prompt engineering experiments with FLAN-T5 on CaseHOLD."""

import os
import json
import argparse
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_loader import load_casehold


LETTERS = ["A", "B", "C", "D", "E"]


def build_zero_shot_prompt(example):
    """Simple zero-shot prompt."""
    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    return (
        f"Choose the correct legal holding for this case.\n\n"
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )


def build_few_shot_prompt(example, few_shot_examples):
    """Few-shot prompt with demonstrations (truncated to avoid overflow)."""
    prompt = "Choose the correct legal holding for each case.\n\n"

    for fs in few_shot_examples:
        # Only show the correct holding to save tokens
        correct_letter = LETTERS[fs["label"]]
        correct_holding = fs["endings"][fs["label"]][:150]
        prompt += (
            f"Context: {fs['context'][:200]}...\n"
            f"Answer: {correct_letter}) {correct_holding}...\n\n---\n\n"
        )

    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    prompt += (
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )
    return prompt


def build_cot_prompt(example):
    """Chain-of-Thought prompt."""
    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    return (
        f"Choose the correct legal holding for this case. "
        f"Think step by step.\n\n"
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Let's think step by step about which holding is correct, then give the answer letter.\n"
        f"Answer:"
    )


def extract_answer(generated_text):
    """Extract the answer letter (A-E) from model output."""
    text = generated_text.strip().upper()
    # Try to find a letter directly
    for letter in LETTERS:
        if text.startswith(letter):
            return letter
    # Search anywhere in the text
    for letter in LETTERS:
        if letter in text:
            return letter
    return "A"  # fallback


def run_experiment(model, tokenizer, dataset, prompt_fn, device, max_samples=None, max_input_length=1024):
    """Run a prompting experiment and return accuracy."""
    preds = []
    labels = []

    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    for example in tqdm(samples, desc="Predicting"):
        prompt = prompt_fn(example)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_letter = extract_answer(generated)
        pred_idx = LETTERS.index(pred_letter) if pred_letter in LETTERS else 0

        preds.append(pred_idx)
        labels.append(example["label"])

    acc = accuracy_score(labels, preds)
    return acc, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--strategy", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot", "all"])
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit test samples (for quick experiments)")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    # Load data
    print("Loading dataset...")
    ds = load_casehold()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Select few-shot examples from train set (fixed seed for reproducibility)
    random.seed(42)
    few_shot_pool = random.sample(range(len(ds["train"])), args.num_shots)
    few_shot_examples = [ds["train"][i] for i in few_shot_pool]

    eval_split = ds[args.split]
    results = {"model": args.model_name, "split": args.split}

    strategies = [args.strategy] if args.strategy != "all" else ["zero_shot", "few_shot", "cot"]

    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} ===")

        if strategy == "zero_shot":
            prompt_fn = build_zero_shot_prompt
        elif strategy == "few_shot":
            prompt_fn = lambda ex: build_few_shot_prompt(ex, few_shot_examples)
        elif strategy == "cot":
            prompt_fn = build_cot_prompt

        acc, preds = run_experiment(
            model, tokenizer, eval_split, prompt_fn, device,
            max_samples=args.max_samples,
        )
        print(f"  Accuracy: {acc:.4f}")
        results[f"{strategy}_accuracy"] = acc

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_name = args.model_name.replace("/", "_")
    results_path = os.path.join(results_dir, f"prompt_{save_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {os.path.basename(results_path)}")


if __name__ == "__main__":
    main()
