"""Preprocessing utilities for CaseHOLD dataset."""

from transformers import AutoTokenizer


def format_for_classification(example):
    """Format a single example as 5 (context, holding) pairs for BERT/RoBERTa.

    Returns dict with:
        - pairs: list of 5 strings "[context] [SEP] [holding]"
        - label: correct index (0-4)
    """
    pairs = [f"{example['context']} [SEP] {h}" for h in example["endings"]]
    return {"pairs": pairs, "label": example["label"]}


def format_for_t5(example):
    """Format a single example as text-to-text for FLAN-T5.

    Returns dict with:
        - input_text: formatted prompt with context + options
        - target_text: correct option letter (A-E)
    """
    letters = ["A", "B", "C", "D", "E"]
    options = "\n".join(f"{letters[i]}) {h}" for i, h in enumerate(example["endings"]))
    input_text = (
        f"Choose the correct holding for this legal case.\n\n"
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )
    target_text = letters[example["label"]]
    return {"input_text": input_text, "target_text": target_text}


def format_for_prompting(example, few_shot_examples=None):
    """Format a single example for LLM prompting (GPT/FLAN-T5).

    Args:
        example: single CaseHOLD example
        few_shot_examples: optional list of examples for few-shot prompting
    """
    letters = ["A", "B", "C", "D", "E"]

    prompt = ""
    if few_shot_examples:
        for fs in few_shot_examples:
            opts = "\n".join(f"{letters[i]}) {h}" for i, h in enumerate(fs["endings"]))
            prompt += (
                f"Context: {fs['context']}\n\n"
                f"Options:\n{opts}\n\n"
                f"Answer: {letters[fs['label']]}\n\n---\n\n"
            )

    options = "\n".join(f"{letters[i]}) {h}" for i, h in enumerate(example["endings"]))
    prompt += (
        f"Context: {example['context']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer:"
    )
    return prompt


def tokenize_for_bert(examples, tokenizer, max_length=256):
    """Tokenize a batch for BERT/RoBERTa multiple-choice classification.

    Each example becomes 5 input sequences (one per candidate holding).
    """
    contexts = []
    holdings = []
    for ctx, ends in zip(examples["context"], examples["endings"]):
        for h in ends:
            contexts.append(ctx)
            holdings.append(h)

    tokenized = tokenizer(
        contexts,
        holdings,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Reshape from (batch*5, seq_len) to (batch, 5, seq_len)
    batch_size = len(examples["context"])
    return {
        k: v.view(batch_size, 5, -1) for k, v in tokenized.items()
    }


if __name__ == "__main__":
    from data_loader import load_casehold

    ds = load_casehold()
    ex = ds["train"][0]

    print("=== Classification format ===")
    clf = format_for_classification(ex)
    for i, p in enumerate(clf["pairs"]):
        print(f"  [{i}] {p[:100]}...")
    print(f"  Label: {clf['label']}")

    print("\n=== T5 format ===")
    t5 = format_for_t5(ex)
    print(f"  Input: {t5['input_text'][:200]}...")
    print(f"  Target: {t5['target_text']}")

    print("\n=== Prompt format ===")
    prompt = format_for_prompting(ex)
    print(f"  {prompt[:200]}...")
