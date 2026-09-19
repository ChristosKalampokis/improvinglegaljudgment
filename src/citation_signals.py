"""Accuracy on test items whose citation carries a contrary-authority signal.

Appendix C observes that signals such as "but see" or "certify conflict with"
mark the cited holding as running against the surrounding prose, and that models
which miss them invert the holding. This script tests that observation: it splits
the test set by whether such a signal appears near the <HOLDING> placeholder and
compares per-model accuracy on the two subsets.

Reads results/predictions.json; writes results/citation_signals.json.
"""

import json
import os
import re

import numpy as np
from scipy.stats import fisher_exact

from data_loader import load_casehold

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Canonical contrary-authority signals (Bluebook). Primary definition.
CONTRARY = [
    r"\bbut see\b",
    r"\bbut cf\b",
    r"\bcontra\b",
    r"\bcertif\w*\s+conflict\b",
]
# Looser set used only as a sensitivity check; "rejecting" and "compare...with"
# often describe what the cited case did rather than signalling disagreement.
CONTRARY_BROAD = CONTRARY + [
    r"\bcompare\b.{0,40}\bwith\b",
    r"\bconflict(?:s|ing)?\s+with\b",
    r"\bdisagree\w*\s+with\b",
    r"\brejecting\b",
]
# Signals that introduce supporting authority, used as a control group.
SUPPORTING = [
    r"\bsee also\b",
    r"\baccord\b",
    r"\bciting\b",
    r"\bquoting\b",
    r"\be\.g\.",
]

WINDOW = 300  # characters before the placeholder to search


def window_before_holding(context):
    i = context.find("<HOLDING>")
    if i < 0:
        i = len(context)
    return context[max(0, i - WINDOW):i].lower()


def matches(text, patterns):
    return [p for p in patterns if re.search(p, text)]


def main():
    with open(os.path.join(RESULTS_DIR, "predictions.json")) as f:
        predictions = json.load(f)
    labels = np.array(predictions.pop("labels"))
    n = len(labels)

    ds = load_casehold()
    test = ds["test"]
    windows = [window_before_holding(test[i]["context"]) for i in range(n)]

    contrary = np.array([bool(matches(w, CONTRARY)) for w in windows])
    supporting = np.array([bool(matches(w, SUPPORTING)) and not c
                           for w, c in zip(windows, contrary)])
    neither = ~contrary & ~supporting

    print(f"test items: {n}")
    print(f"  contrary-signal:   {contrary.sum():4d} ({contrary.mean()*100:.1f}%)")
    print(f"  supporting-signal: {supporting.sum():4d} ({supporting.mean()*100:.1f}%)")
    print(f"  neither:           {neither.sum():4d} ({neither.mean()*100:.1f}%)")

    # which contrary cues actually fire, for reporting
    counts = {}
    for w in windows:
        for p in matches(w, CONTRARY):
            counts[p] = counts.get(p, 0) + 1
    print("\ncontrary cue frequencies:")
    for p, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {p:35s} {c}")

    out = {
        "n_test": int(n),
        "window_chars": WINDOW,
        "n_contrary": int(contrary.sum()),
        "n_supporting": int(supporting.sum()),
        "n_neither": int(neither.sum()),
        "cue_counts": {p: int(c) for p, c in counts.items()},
        "models": {},
    }

    broad = np.array([bool(matches(w, CONTRARY_BROAD)) for w in windows])
    out["n_contrary_broad"] = int(broad.sum())

    print(f"\n{'Model':<25} {'contrary':>9} {'other':>9} {'gap':>8} {'p':>8} {'gap(broad)':>11}")
    print("-" * 74)
    n_neg = 0
    for name, preds in predictions.items():
        preds = np.array(preds[:n])
        correct = preds == labels
        other = ~contrary
        a_c, a_o = correct[contrary].mean(), correct[other].mean()
        table = [[int(correct[contrary].sum()), int((~correct[contrary]).sum())],
                 [int(correct[other].sum()), int((~correct[other]).sum())]]
        p = float(fisher_exact(table)[1])
        gap = a_c - a_o
        gap_b = correct[broad].mean() - correct[~broad].mean()
        if name != "SBERT" and gap < 0:
            n_neg += 1
        out["models"][name] = {
            "acc_contrary": round(float(a_c), 4),
            "acc_other": round(float(a_o), 4),
            "acc_supporting": round(float(correct[supporting].mean()), 4),
            "acc_neither": round(float(correct[neither].mean()), 4),
            "gap_contrary_minus_other": round(float(gap), 4),
            "gap_broad": round(float(gap_b), 4),
            "fisher_p": p,
        }
        print(f"{name:<25} {a_c:>9.4f} {a_o:>9.4f} {gap:>+8.4f} {p:>8.3f} {gap_b:>+11.4f}")

    out["n_trained_models_with_negative_gap"] = n_neg
    print(f"\ntrained models (excluding SBERT) with a negative gap: {n_neg} of 6")

    path = os.path.join(RESULTS_DIR, "citation_signals.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
