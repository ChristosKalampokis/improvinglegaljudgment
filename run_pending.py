"""Sequential launcher for the long experiment queue (one GPU).

Steps are skipped if their results already exist, so the launcher can be
re-run after any interruption. Meant to be started via Task Scheduler so it
survives parent-process restarts. Progress goes to logs/pending.log; a
pending_done marker is written at the end.
"""
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT, "src")
RESULTS = os.path.join(PROJECT, "results")
LOGS = os.path.join(PROJECT, "logs")
PLOG = os.path.join(LOGS, "pending.log")


def log(msg):
    with open(PLOG, "a") as f:
        f.write(msg + "\n")


def run_step(name, args, cwd, done_check):
    if done_check():
        log(f"SKIP {name} (output already exists)")
        return True
    log(f"START {name}")
    with open(os.path.join(LOGS, f"pend_{name}.log"), "w") as f:
        rc = subprocess.run([sys.executable] + args, cwd=cwd,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END {name} rc={rc}")
    return rc == 0


STEPS = [
    ("rag_lora",
     ["rag_pipeline.py", "--llm_model", "google/flan-t5-base",
      "--lora_adapter", "../results/google_flan-t5-base_lora_best",
      "--k_values", "0,1,3,5"],
     SRC,
     lambda: os.path.exists(os.path.join(RESULTS, "rag_google_flan-t5-base_lora_results.json"))),
    ("rag_holdix",
     ["rag_pipeline.py", "--llm_model", "google/flan-t5-base",
      "--index_holdings", "--k_values", "1,3,5"],
     SRC,
     lambda: os.path.exists(os.path.join(RESULTS, "rag_google_flan-t5-base_holdix_results.json"))),
    ("lora_large",
     ["finetune_lora.py", "--model_name", "google/flan-t5-large",
      "--epochs", "3", "--batch_size", "2", "--grad_accum", "2",
      "--bf16", "--gradient_checkpointing"],
     SRC,
     lambda: os.path.exists(os.path.join(RESULTS, "lora_google_flan-t5-large_results.json"))),
    # --- FLAN-T5-large reruns of the base-model experiments ---
    ("prompt_large",
     ["prompt_engineering.py", "--model_name", "google/flan-t5-large", "--strategy", "all"],
     SRC,
     lambda: os.path.exists(os.path.join(RESULTS, "prompt_google_flan-t5-large_results.json"))),
    ("rag_large",
     ["rag_pipeline.py", "--llm_model", "google/flan-t5-large", "--k_values", "0,1,3,5"],
     SRC,
     lambda: os.path.exists(os.path.join(RESULTS, "rag_google_flan-t5-large_results.json"))),
    ("rag_lora_large",
     ["rag_pipeline.py", "--llm_model", "google/flan-t5-large",
      "--lora_adapter", "../results/google_flan-t5-large_lora_best",
      "--k_values", "0,1,3,5"],
     SRC,
     lambda: os.path.exists(os.path.join(RESULTS, "rag_google_flan-t5-large_lora_results.json"))),
    ("multiseed",
     ["run_multiseed.py"],
     PROJECT,
     lambda: False),  # run_multiseed has its own per-(model,seed) skip logic
]


def main():
    os.makedirs(LOGS, exist_ok=True)
    all_ok = True
    for name, args, cwd, done_check in STEPS:
        if not run_step(name, args, cwd, done_check):
            all_ok = False
            log(f"FAILED {name} - continuing with remaining steps")

    marker = "pending_done.marker" if all_ok else "pending_done_with_errors.marker"
    with open(os.path.join(LOGS, marker), "w") as f:
        f.write("done")


if __name__ == "__main__":
    main()
