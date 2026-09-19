"""Multi-seed reruns for variance estimation.

Retrains the encoders and the base LoRA model with extra seeds so results
can be reported as mean +/- std. The primary (unseeded) runs count as the
first sample. Completed (model, seed) combos are skipped, so the script is
safe to interrupt and relaunch. Each encoder run takes ~3.5h on an RTX 3070.

Usage:
    python run_multiseed.py                # run whatever is missing
    python run_multiseed.py --summarize    # aggregate existing results only
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT, "src")
RESULTS = os.path.join(PROJECT, "results")
LOGS = os.path.join(PROJECT, "logs")

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "nlpaueb/legal-bert-base-uncased",
    "casehold/custom-legalbert",
]
# Base LoRA only: large costs ~14h/seed to defend an uncontested ~14-point
# lead, and its zero-init adapters make runs start from the same function.
LORA_MODELS = [
    "google/flan-t5-base",
]
SEEDS = [123, 2026]


def result_path(model, seed=None):
    name = model.replace("/", "_")
    if seed is not None:
        name += f"_seed{seed}"
    return os.path.join(RESULTS, f"{name}_results.json")


def lora_result_path(model, seed=None):
    name = model.replace("/", "_")
    if seed is not None:
        name += f"_seed{seed}"
    return os.path.join(RESULTS, f"lora_{name}_results.json")


def checkpoint_path(model, seed):
    name = model.replace("/", "_") + f"_seed{seed}"
    return os.path.join(RESULTS, f"{name}_best.pt")


def show_row(label, paths):
    accs = []
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                accs.append(json.load(f)["test_accuracy"])
    if not accs:
        return
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    acc_str = ", ".join(f"{a:.4f}" for a in accs)
    print(f"{label:<40} {len(accs):>3} {mean:>8.4f} {std:>8.4f}  [{acc_str}]")


def summarize():
    print(f"\n{'Model':<40} {'n':>3} {'mean':>8} {'std':>8}  accuracies")
    print("-" * 85)
    for model in MODELS:
        show_row(model, [result_path(model, s) for s in [None] + SEEDS])
    for model in LORA_MODELS:
        show_row(f"LoRA {model}", [lora_result_path(model, s) for s in [None] + SEEDS])
    print("\n(n=1 rows have only the primary run; seed runs not done yet)")


def run(cmd, log_name):
    with open(os.path.join(LOGS, log_name), "w") as f:
        return subprocess.run([sys.executable] + cmd, cwd=SRC,
                              stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true",
                        help="Only aggregate existing results, run nothing")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep seed-run checkpoints (~440 MB each)")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        return

    os.makedirs(LOGS, exist_ok=True)
    for model in MODELS:
        for seed in SEEDS:
            if os.path.exists(result_path(model, seed)):
                print(f"SKIP {model} seed={seed} (results exist)")
                continue
            log_name = f"multiseed_{model.replace('/', '_')}_seed{seed}.log"
            print(f"RUN  {model} seed={seed} -> logs/{log_name}")
            rc = run(["baseline_bert.py", "--model_name", model,
                      "--epochs", "3", "--batch_size", "4", "--seed", str(seed)],
                     log_name)
            print(f"     exit code {rc}")

            ckpt = checkpoint_path(model, seed)
            if rc == 0 and not args.keep_checkpoints and os.path.exists(ckpt):
                os.remove(ckpt)
                print(f"     deleted seed checkpoint ({os.path.basename(ckpt)})")

    # LoRA adapters are only ~7 MB, so they are kept
    for model in LORA_MODELS:
        for seed in SEEDS:
            if os.path.exists(lora_result_path(model, seed)):
                print(f"SKIP LoRA {model} seed={seed} (results exist)")
                continue
            log_name = f"multiseed_lora_{model.replace('/', '_')}_seed{seed}.log"
            print(f"RUN  LoRA {model} seed={seed} -> logs/{log_name}")
            rc = run(["finetune_lora.py", "--model_name", model,
                      "--epochs", "3", "--batch_size", "4", "--seed", str(seed)],
                     log_name)
            print(f"     exit code {rc}")

    summarize()


if __name__ == "__main__":
    main()
