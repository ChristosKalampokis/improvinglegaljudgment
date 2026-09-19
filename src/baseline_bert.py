"""BERT/RoBERTa baseline for CaseHOLD multiple-choice classification."""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_loader import load_casehold


class CaseHOLDDataset(Dataset):
    """Tokenizes CaseHOLD examples for multiple-choice classification."""

    def __init__(self, split, tokenizer, max_length=256):
        self.data = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        context = example["context"]
        endings = example["endings"]
        label = example["label"]

        # Tokenize each (context, holding) pair
        input_ids_list = []
        attention_mask_list = []
        for holding in endings:
            encoded = self.tokenizer(
                context,
                holding,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids_list.append(encoded["input_ids"].squeeze(0))
            attention_mask_list.append(encoded["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(input_ids_list),        # (5, max_length)
            "attention_mask": torch.stack(attention_mask_list),  # (5, max_length)
            "label": torch.tensor(label, dtype=torch.long),
        }


class MultipleChoiceModel(nn.Module):
    """Encoder + linear head for 5-way multiple choice."""

    def __init__(self, model_name, **encoder_kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, **encoder_kwargs)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch, 5, seq_len)
        batch_size, num_choices, seq_len = input_ids.shape

        # Flatten to (batch*5, seq_len)
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        logits = self.classifier(cls_output)  # (batch*5, 1)
        logits = logits.view(batch_size, num_choices)  # (batch, 5)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; adds a _seed{N} suffix to output filenames")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    # Load data
    print("Loading dataset...")
    ds = load_casehold()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = CaseHOLDDataset(ds["train"], tokenizer, args.max_length)
    val_dataset = CaseHOLDDataset(ds["validation"], tokenizer, args.max_length)
    test_dataset = CaseHOLDDataset(ds["test"], tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, num_workers=0)

    # Model
    model = MultipleChoiceModel(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop
    best_val_acc = 0
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_name = args.model_name.replace("/", "_")
    if args.seed is not None:
        save_name += f"_seed{args.seed}"

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, f"{save_name}_best.pt"))
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    # Test evaluation
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(results_dir, f"{save_name}_best.pt"), weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Save results
    results = {
        "model": args.model_name,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    results_path = os.path.join(results_dir, f"{save_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    # Print only the basename: the full project path contains Greek characters
    # that crash on cp1252 consoles (e.g. when run via Task Scheduler).
    print(f"Results saved to: {os.path.basename(results_path)}")


if __name__ == "__main__":
    main()
