"""Master script to run all experiments sequentially.

Usage:
    python run_all.py              # Run everything
    python run_all.py --quick      # Quick test with 100 samples per experiment
"""

import subprocess
import sys
import argparse


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable] + cmd, cwd=".")
    if result.returncode != 0:
        print(f"  WARNING: {description} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test with limited samples")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_prompting", action="store_true")
    parser.add_argument("--skip_rag", action="store_true")
    parser.add_argument("--skip_lora", action="store_true")
    args = parser.parse_args()

    max_samples = ["--max_samples", "100"] if args.quick else []

    # Phase 3: Baselines
    if not args.skip_baselines:
        run(["baseline_bert.py", "--model_name", "bert-base-uncased", "--epochs", "3", "--batch_size", "4"],
            "Phase 3a: BERT Baseline")

        run(["baseline_bert.py", "--model_name", "roberta-base", "--epochs", "3", "--batch_size", "4"],
            "Phase 3b: RoBERTa Baseline")

        run(["baseline_sbert.py"],
            "Phase 3c: SBERT Baseline")

        run(["baseline_bert.py", "--model_name", "casehold/custom-legalbert", "--epochs", "3", "--batch_size", "4"],
            "Phase 3d: Legal-BERT Baseline (CaseHOLD custom)")

        run(["baseline_bert.py", "--model_name", "nlpaueb/legal-bert-base-uncased", "--epochs", "3", "--batch_size", "4"],
            "Phase 3e: Legal-BERT Baseline (NLPAUEB)")

    # Phase 4: Prompt Engineering
    if not args.skip_prompting:
        run(["prompt_engineering.py", "--model_name", "google/flan-t5-base",
             "--strategy", "all"] + max_samples,
            "Phase 4: Prompt Engineering (FLAN-T5-base)")

    # Phase 5: RAG
    if not args.skip_rag:
        run(["rag_pipeline.py", "--llm_model", "google/flan-t5-base",
             "--k_values", "0,1,3,5"] + max_samples,
            "Phase 5: RAG Pipeline Ablation")

    # Phase 6: LoRA Fine-tuning
    if not args.skip_lora:
        run(["finetune_lora.py", "--model_name", "google/flan-t5-base",
             "--epochs", "3", "--batch_size", "4", "--lora_r", "16"],
            "Phase 6: LoRA Fine-tuning (FLAN-T5-base)")

    # Compile results
    run(["evaluate_all.py"], "Compile All Results")


if __name__ == "__main__":
    main()
