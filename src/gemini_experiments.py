"""Gemini API experiments (prompt engineering + RAG) for CaseHOLD.

Supports two backends:
  1. AI Studio (free tier): --api_key YOUR_KEY
  2. Vertex AI ($300 credits): --vertex_ai --project YOUR_PROJECT --location REGION

Features:
  - Resume logic: skips already-completed experiments
  - Incremental saving: results saved after each experiment
"""

import os
import json
import argparse
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_loader import load_casehold


LETTERS = ["A", "B", "C", "D", "E"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ============================================================
# Prompt builders (same logic as FLAN-T5, but no truncation needed)
# ============================================================

def build_zero_shot_prompt(example):
    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    return (
        f"Choose the correct legal holding for this case. "
        f"Reply with ONLY the letter (A, B, C, D, or E).\n\n"
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )


def build_few_shot_prompt(example, few_shot_examples):
    prompt = "Choose the correct legal holding for each case. Reply with ONLY the letter.\n\n"
    for fs in few_shot_examples:
        correct_letter = LETTERS[fs["label"]]
        options_fs = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(fs["endings"]))
        prompt += (
            f"Context: {fs['context']}\n\n"
            f"Options:\n{options_fs}\n\n"
            f"Answer: {correct_letter}\n\n---\n\n"
        )
    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    prompt += (
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )
    return prompt


def build_cot_prompt(example):
    options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
    return (
        f"Choose the correct legal holding for this case. "
        f"Think step by step, then give your final answer as a single letter (A-E).\n\n"
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Let's analyze each option step by step:\n"
    )


def build_rag_prompt(example, retrieved_examples):
    prompt = (
        "Choose the correct legal holding for the case. "
        "Here are similar resolved cases for reference. "
        "Reply with ONLY the letter (A, B, C, D, or E).\n\n"
    )
    for i, ret in enumerate(retrieved_examples):
        correct_holding = ret["endings"][ret["label"]]
        prompt += (
            f"Similar case {i+1}:\n"
            f"Context: {ret['context']}\n"
            f"Correct holding: {LETTERS[ret['label']]}) {correct_holding}\n\n"
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


# ============================================================
# Answer extraction
# ============================================================

def extract_answer(text):
    """Extract answer letter from Gemini response."""
    import re
    text = text.strip()

    # 1) Short response (just a letter, maybe with punctuation) — direct answer
    if len(text) <= 3:
        first = text.upper()[0] if text else ""
        if first in LETTERS:
            return first

    # 2) CoT / long response — look for final-answer patterns
    #    Search from the END of the text for explicit answer markers
    patterns = [
        r'\\boxed\{([A-Ea-e])\}',           # LaTeX boxed like \boxed{E}
        r'(?:final answer|the answer|my answer|correct answer|best answer)\s*(?:is|:)\s*\(?([A-Ea-e])\)?',
        r'(?:I (?:would |will )?(?:choose|select|pick|go with))\s*\(?([A-Ea-e])\)?',
        r'\*\*([A-Ea-e])\)\*\*',            # bold option like **C)**
        r'\*\*([A-Ea-e])\*\*',              # bold letter like **C**
        r'\b([A-Ea-e])\)?\s*$',             # letter at end of text
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Return the first non-None captured group from the last match
            for g in matches[-1].groups():
                if g:
                    return g.upper()

    # 3) Fallback for short direct responses: check first character
    upper = text.upper()
    if upper[0] in LETTERS:
        return upper[0]

    return "A"  # last resort fallback


# ============================================================
# Gemini API call with rate-limit handling (new google-genai SDK)
# ============================================================

def call_gemini(client, model_name, prompt, max_retries=5, max_output_tokens=256,
                disable_thinking=False):
    """Call Gemini API with retry logic for rate limits."""
    from google.genai import types

    config_kwargs = dict(
        max_output_tokens=max_output_tokens,
        temperature=0.0,
    )
    if disable_thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            # Extract text from response
            if response.text:
                return response.text
            # If blocked or empty, return fallback
            return "A"
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                wait = min(30 * (2 ** attempt), 120)
                print(f"\n  Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"\n  API error: {error_str[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return "A"
    return "A"


# ============================================================
# RAG retrieval (reuse FAISS from rag_pipeline)
# ============================================================

def build_retrieval_index(train_data, sbert_model, batch_size=64):
    """Build FAISS index over training contexts."""
    import faiss

    print("Building FAISS retrieval index...")
    contexts = train_data["context"]
    embeddings = sbert_model.encode(contexts, batch_size=batch_size, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index: {index.ntotal} vectors (dim={dim})")
    return index


def retrieve_similar(query_text, sbert_model, index, train_data, k=3):
    """Retrieve top-k similar training examples."""
    import faiss

    query_emb = sbert_model.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)
    return [train_data[int(idx)] for idx in indices[0]]


# ============================================================
# Results loading / saving with resume support
# ============================================================

def _get_results_path(model_name):
    """Get the path for results JSON file."""
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return os.path.join(RESULTS_DIR, f"gemini_{safe_name}_results.json")


def _load_existing_results(model_name):
    """Load existing results if available (for resume)."""
    path = _get_results_path(model_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_results(results, model_name):
    """Save results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _get_results_path(model_name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ============================================================
# Main experiment runner
# ============================================================

def run_experiment(client, model_name, dataset, prompt_fn, max_samples=None, delay=0.5,
                   max_output_tokens=256, disable_thinking=False):
    """Run prompting experiment via Gemini API."""
    preds = []
    labels = []

    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    for example in tqdm(samples, desc="Gemini"):
        prompt = prompt_fn(example)
        response_text = call_gemini(client, model_name, prompt,
                                    max_output_tokens=max_output_tokens,
                                    disable_thinking=disable_thinking)
        pred_letter = extract_answer(response_text)
        pred_idx = LETTERS.index(pred_letter) if pred_letter in LETTERS else 0

        preds.append(pred_idx)
        labels.append(example["label"])

        time.sleep(delay)  # respect rate limits

    acc = accuracy_score(labels, preds)
    return acc, preds


def create_client(args):
    """Create a google-genai client based on CLI arguments."""
    from google import genai

    if args.vertex_ai:
        # Vertex AI mode — uses Application Default Credentials (gcloud auth)
        # Charges go to the GCP project ($300 credits)
        print(f"Backend: Vertex AI (project={args.project}, location={args.location})")
        client = genai.Client(
            vertexai=True,
            project=args.project,
            location=args.location,
        )
    else:
        # AI Studio mode — uses API key (free tier)
        print(f"Backend: AI Studio (API key)")
        client = genai.Client(api_key=args.api_key)

    return client


def main():
    parser = argparse.ArgumentParser(description="Gemini API experiments on CaseHOLD")

    # Backend selection
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--api_key", type=str, help="AI Studio API key (free tier)")
    backend.add_argument("--vertex_ai", action="store_true",
                         help="Use Vertex AI with GCP credits (requires gcloud auth)")

    # Vertex AI options
    parser.add_argument("--project", type=str, default="project-c827f05c-25dd-4205-b86",
                        help="GCP project ID (for Vertex AI)")
    parser.add_argument("--location", type=str, default="us-central1",
                        help="GCP region (for Vertex AI)")

    # Model and experiment options
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
                        help="Gemini model to use")
    parser.add_argument("--experiments", type=str, default="all",
                        choices=["zero_shot", "few_shot", "cot", "rag", "all"],
                        help="Which experiments to run")
    parser.add_argument("--rag_k_values", type=str, default="1,3,5",
                        help="Comma-separated k values for RAG")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit test samples (for quick tests)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between API calls in seconds")
    parser.add_argument("--num_shots", type=int, default=3)
    args = parser.parse_args()

    # Create client
    client = create_client(args)
    model_name = args.model
    print(f"Model: {model_name}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Delay between calls: {args.delay}s")

    # Load existing results for resume
    existing = _load_existing_results(model_name)
    if existing:
        print(f"\nFound existing results — will skip completed experiments:")
        for key in existing:
            if "accuracy" in key:
                print(f"  {key}: {existing[key]}")

    # Test API connectivity
    print("\nTesting API connection...")
    try:
        test_resp = client.models.generate_content(
            model=model_name,
            contents="Say hello in one word.",
        )
        print(f"  API OK: {test_resp.text.strip()[:50]}")
    except Exception as e:
        print(f"  API ERROR: {e}")
        return

    # Load dataset
    print("\nLoading dataset...")
    ds = load_casehold()
    eval_split = ds["test"]

    # Fixed few-shot examples
    random.seed(42)
    few_shot_pool = random.sample(range(len(ds["train"])), args.num_shots)
    few_shot_examples = [ds["train"][i] for i in few_shot_pool]

    # Start from existing results (for resume)
    results = {
        "model": model_name,
        "backend": "vertex_ai" if args.vertex_ai else "ai_studio",
        "max_samples": args.max_samples,
    }
    results.update(existing)

    # Determine which experiments to run
    if args.experiments == "all":
        experiments = ["zero_shot", "few_shot", "cot", "rag"]
    else:
        experiments = [args.experiments]

    # ---- Prompting experiments ----
    for exp in experiments:
        if exp == "rag":
            continue  # handle RAG separately below

        result_key = f"{exp}_accuracy"
        if result_key in existing:
            print(f"\n  SKIPPING {exp} (already done: {existing[result_key]})")
            continue

        print(f"\n{'='*50}")
        print(f"  Experiment: {exp} ({model_name})")
        print(f"{'='*50}")

        if exp == "zero_shot":
            prompt_fn = build_zero_shot_prompt
        elif exp == "few_shot":
            prompt_fn = lambda ex: build_few_shot_prompt(ex, few_shot_examples)
        elif exp == "cot":
            prompt_fn = build_cot_prompt

        # CoT needs more output tokens for the visible reasoning.
        # Flash supports thinking_budget=0 so we disable its built-in
        # thinking (which eats the output token budget).
        # Pro does NOT support thinking_budget=0, so we keep thinking
        # enabled but use a larger token budget to accommodate both.
        is_cot = (exp == "cot")
        is_pro = ("pro" in model_name.lower())
        if is_cot:
            cot_tokens = 8192 if is_pro else 2048
            cot_disable = False if is_pro else True
        else:
            cot_tokens = 256
            cot_disable = False
        acc, preds = run_experiment(
            client, model_name, eval_split, prompt_fn,
            max_samples=args.max_samples, delay=args.delay,
            max_output_tokens=cot_tokens,
            disable_thinking=cot_disable,
        )
        print(f"  {exp} accuracy: {acc:.4f} ({int(acc * len(preds))}/{len(preds)})")
        results[f"{exp}_accuracy"] = round(acc, 4)

        # Save incrementally
        _save_results(results, model_name)

    # ---- RAG experiments ----
    if "rag" in experiments:
        k_values = [int(k) for k in args.rag_k_values.split(",")]

        # Check if any RAG experiments still need to run
        rag_needed = [k for k in k_values if f"rag_k{k}_accuracy" not in existing]
        if not rag_needed:
            print(f"\n  SKIPPING all RAG experiments (already done)")
        else:
            from sentence_transformers import SentenceTransformer

            print(f"\n{'='*50}")
            print(f"  Loading SBERT for RAG retrieval...")
            print(f"{'='*50}")
            sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            index = build_retrieval_index(ds["train"], sbert_model)

            for k in k_values:
                result_key = f"rag_k{k}_accuracy"
                if result_key in existing:
                    print(f"\n  SKIPPING RAG k={k} (already done: {existing[result_key]})")
                    continue

                print(f"\n--- RAG k={k} ({model_name}) ---")

                def rag_prompt_fn(ex, _k=k):
                    retrieved = retrieve_similar(ex["context"], sbert_model, index, ds["train"], k=_k)
                    return build_rag_prompt(ex, retrieved)

                acc, preds = run_experiment(
                    client, model_name, eval_split, rag_prompt_fn,
                    max_samples=args.max_samples, delay=args.delay,
                )
                print(f"  RAG k={k} accuracy: {acc:.4f} ({int(acc * len(preds))}/{len(preds)})")
                results[f"rag_k{k}_accuracy"] = round(acc, 4)

                _save_results(results, model_name)

    # Final save
    _save_results(results, model_name)
    print(f"\n{'='*50}")
    print("  ALL DONE! Results summary:")
    print(f"{'='*50}")
    for key, val in results.items():
        if "accuracy" in key:
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
