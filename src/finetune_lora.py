"""LoRA/QLoRA fine-tuning of FLAN-T5 on CaseHOLD."""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_loader import load_casehold


LETTERS = ["A", "B", "C", "D", "E"]


class CaseHOLDT5Dataset(Dataset):
    """Format CaseHOLD for text-to-text FLAN-T5 training."""

    def __init__(self, split, tokenizer, max_input_length=512, max_target_length=8):
        self.data = split
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        options = "\n".join(f"{LETTERS[i]}) {h}" for i, h in enumerate(example["endings"]))
        input_text = (
            f"Choose the correct holding for this legal case.\n\n"
            f"Context: {example['context']}\n\n"
            f"Options:\n{options}\n\n"
            f"Answer:"
        )
        target_text = LETTERS[example["label"]]

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token ids with -100 so they're ignored in loss
        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1):
    model.train()
    total_loss = 0

    optimizer.zero_grad()
    step = -1
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

    # Flush a trailing partial accumulation window
    if grad_accum > 1 and (step + 1) % grad_accum != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in decoded:
            text = text.strip().upper()
            pred = 0
            for i, letter in enumerate(LETTERS):
                if text.startswith(letter):
                    pred = i
                    break
            all_preds.append(pred)

        # Recover true labels
        for label_ids in labels:
            label_ids = label_ids[label_ids != -100]
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True).strip().upper()
            true_idx = 0
            for i, letter in enumerate(LETTERS):
                if label_text.startswith(letter):
                    true_idx = i
                    break
            all_labels.append(true_idx)

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples (for quick tests)")
    parser.add_argument("--bf16", action="store_true",
                        help="Load the base model in bfloat16 (needed for flan-t5-large on 8GB)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for activation memory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; adds a _seed{N} suffix to output filenames")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    # Load data
    print("Loading dataset...")
    ds = load_casehold()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]
    if args.max_samples:
        train_data = train_data[:args.max_samples]
        val_data = val_data[:min(args.max_samples // 4, len(val_data))]
        test_data = test_data[:min(args.max_samples // 4, len(test_data))]
        print(f"Quick test mode: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

    train_dataset = CaseHOLDT5Dataset(train_data, tokenizer)
    val_dataset = CaseHOLDT5Dataset(val_data, tokenizer)
    test_dataset = CaseHOLDT5Dataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, num_workers=0)

    # Load model (optionally quantized)
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, quantization_config=bnb_config)
    else:
        # bf16 rather than fp16: fp16 is numerically unstable with T5
        dtype = torch.bfloat16 if args.bf16 else None
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, dtype=dtype).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed for gradients to reach the LoRA layers

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(train_loader) // args.grad_accum)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_name = args.model_name.replace("/", "_")
    if args.seed is not None:
        save_name += f"_seed{args.seed}"
    best_val_acc = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device,
                                 grad_accum=args.grad_accum)
        print(f"Train Loss: {train_loss:.4f}")

        val_acc = evaluate(model, val_loader, tokenizer, device)
        print(f"Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(results_dir, f"{save_name}_lora_best"))
            print(f"  Saved best LoRA adapter (val_acc={val_acc:.4f})")

    # Test evaluation
    print("\n--- Test Evaluation ---")
    test_acc = evaluate(model, test_loader, tokenizer, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    results = {
        "model": args.model_name,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "use_4bit": args.use_4bit,
        "bf16": args.bf16,
        "grad_accum": args.grad_accum,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
    }
    results_path = os.path.join(results_dir, f"lora_{save_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {os.path.basename(results_path)}")


if __name__ == "__main__":
    main()
