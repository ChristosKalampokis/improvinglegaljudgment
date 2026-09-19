"""Significance tests for the model comparisons.

Pairwise McNemar tests between all local models (from predictions.json) and
bootstrap 95% CIs per model. Gemini gets normal-approximation CIs only, since
the API runs kept no per-sample predictions. Writes results/significance.json.
"""

import glob
import json
import os

import numpy as np
from scipy.stats import binomtest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_BOOT = 10_000
CHUNK = 1_000
RNG = np.random.default_rng(42)


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness vectors."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return b, c, float(p)


def bootstrap_ci(correct, n_boot=N_BOOT):
    """Percentile bootstrap 95% CI for accuracy."""
    n = len(correct)
    means = np.empty(n_boot)
    done = 0
    while done < n_boot:
        size = min(CHUNK, n_boot - done)
        idx = RNG.integers(0, n, size=(size, n))
        means[done:done + size] = correct[idx].mean(axis=1)
        done += size
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def normal_ci(acc, n):
    se = (acc * (1 - acc) / n) ** 0.5
    return acc - 1.96 * se, acc + 1.96 * se


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    with open(pred_path) as f:
        predictions = json.load(f)
    labels = np.array(predictions.pop("labels"))
    n = len(labels)

    correct = {name: (np.array(preds[:n]) == labels)
               for name, preds in predictions.items()}
    models = list(correct.keys())

    out = {"n_samples": n, "n_bootstrap": N_BOOT,
           "note": ("Gemini models: normal-approximation CIs only; McNemar requires "
                    "per-sample predictions, which the API runs did not retain."),
           "bootstrap_ci_95": {}, "mcnemar": []}

    print(f"\n95% bootstrap confidence intervals ({N_BOOT} resamples, n={n}):")
    print(f"{'Model':<25} {'Acc':>8} {'CI low':>8} {'CI high':>8}")
    print("-" * 55)
    for name in models:
        acc = float(correct[name].mean())
        lo, hi = bootstrap_ci(correct[name])
        out["bootstrap_ci_95"][name] = {"accuracy": round(acc, 4),
                                        "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        print(f"{name:<25} {acc:>8.4f} {lo:>8.4f} {hi:>8.4f}")

    print(f"\nGemini 95% CIs (normal approximation, n={n}):")
    out["gemini_normal_ci_95"] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "gemini_*_results.json"))):
        with open(path) as f:
            g = json.load(f)
        model = g.get("model", os.path.basename(path))
        for key, val in g.items():
            if key.endswith("_accuracy"):
                lo, hi = normal_ci(val, n)
                tag = f"{model}/{key.replace('_accuracy', '')}"
                out["gemini_normal_ci_95"][tag] = {"accuracy": val,
                                                   "ci_low": round(lo, 4),
                                                   "ci_high": round(hi, 4)}
                print(f"  {tag:<40} {val:.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\nPairwise McNemar tests (exact two-sided binomial):")
    print(f"{'Model A':<25} {'Model B':<25} {'b':>5} {'c':>5} {'p-value':>12}  sig")
    print("-" * 85)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, bn = models[i], models[j]
            b, c, p = mcnemar(correct[a], correct[bn])
            out["mcnemar"].append({"model_a": a, "model_b": bn,
                                   "a_right_b_wrong": b, "b_right_a_wrong": c,
                                   "p_value": p, "significance": stars(p)})
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"{a:<25} {bn:<25} {b:>5} {c:>5} {p_str:>12}  {stars(p)}")

    out_path = os.path.join(RESULTS_DIR, "significance.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {os.path.basename(out_path)}")
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


if __name__ == "__main__":
    main()
