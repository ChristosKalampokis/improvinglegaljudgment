"""Compile all experiment results into a summary table."""

import os
import json
import glob


def load_all_results(results_dir):
    """Load all JSON result files from the results directory."""
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        with open(path) as f:
            data = json.load(f)
            data["_file"] = os.path.basename(path)
            results.append(data)
    return results


def print_summary_table(results):
    """Print a formatted summary table of all experiments."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model/Experiment':<45} {'Test Acc':>10} {'Val Acc':>10}")
    print("-" * 80)

    for r in results:
        name = r.get("model", r.get("llm_model", "unknown"))
        file = r["_file"]

        # Determine experiment type from filename
        if file.startswith("sbert"):
            name = f"SBERT ({name})"
        elif file.startswith("prompt_"):
            # Show all strategy results
            for key in r:
                if key.endswith("_accuracy") and key != "test_accuracy":
                    strategy = key.replace("_accuracy", "")
                    print(f"  Prompt/{strategy} ({name})"[:44].ljust(45)
                          + f"{r[key]:>10.4f}" + f"{'':>10}")
            continue
        elif file.startswith("rag_"):
            retr = r.get("sbert_model", "")
            retr_tag = f", retr={retr.split('/')[-1]}" if retr and retr != "all-MiniLM-L6-v2" else ""
            for key in r:
                if key.startswith("k") and key.endswith("_accuracy"):
                    k_val = key.replace("_accuracy", "")
                    print(f"  RAG {k_val} ({name}{retr_tag})"[:44].ljust(45)
                          + f"{r[key]:>10.4f}" + f"{'':>10}")
            continue
        elif file.startswith("gemini_"):
            # Gemini files hold multiple experiment accuracies, one per key
            for key in r:
                if key.endswith("_accuracy"):
                    exp = key.replace("_accuracy", "")
                    print(f"  Gemini/{exp} ({name})"[:44].ljust(45)
                          + f"{r[key]:>10.4f}" + f"{'':>10}")
            continue
        elif file.startswith("lora_"):
            name = f"LoRA ({name}, r={r.get('lora_r', '?')})"

        test_acc = r.get("test_accuracy", r.get("test_accuracy", ""))
        val_acc = r.get("best_val_accuracy", r.get("validation_accuracy", ""))

        test_str = f"{test_acc:.4f}" if isinstance(test_acc, float) else str(test_acc)
        val_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else str(val_acc)

        print(f"  {name[:43]:<45}{test_str:>10}{val_str:>10}")

    print("-" * 80)
    print(f"  {'Random baseline':<45}{'0.2000':>10}")
    print("=" * 80)


def save_summary_json(results, results_dir):
    """Save a consolidated summary JSON."""
    summary = {}
    for r in results:
        name = r["_file"].replace("_results.json", "")
        summary[name] = {k: v for k, v in r.items() if k != "_file"}

    path = os.path.join(results_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {os.path.basename(path)}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    if not os.path.exists(results_dir):
        print("No results directory found.")
        return

    results = load_all_results(results_dir)
    if not results:
        print("No result files found.")
        return

    print(f"Found {len(results)} result files.")
    print_summary_table(results)
    save_summary_json(results, results_dir)


if __name__ == "__main__":
    main()
