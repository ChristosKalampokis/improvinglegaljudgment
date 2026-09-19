"""Re-runs the whole evaluation stack (error analysis, semantic score,
significance, summary, figures) so every artifact covers the full model set.
Intentionally overwrites previous analysis outputs.
"""
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT, "src")
LOGS = os.path.join(PROJECT, "logs")
PLOG = os.path.join(LOGS, "final_analysis.log")

STEPS = [
    ("error_analysis", ["error_analysis.py"]),
    ("semantic_score", ["semantic_score.py"]),
    ("significance", ["significance.py"]),
    ("evaluate_all", ["evaluate_all.py"]),
    ("visualize", ["visualize_results.py"]),
]


def log(msg):
    with open(PLOG, "a") as f:
        f.write(msg + "\n")


def main():
    os.makedirs(LOGS, exist_ok=True)
    all_ok = True
    for name, args in STEPS:
        log(f"START {name}")
        with open(os.path.join(LOGS, f"final_{name}.log"), "w") as f:
            rc = subprocess.run([sys.executable] + args, cwd=SRC,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"END {name} rc={rc}")
        if rc != 0:
            all_ok = False
            log(f"FAILED {name} - continuing")

    marker = "final_analysis_done.marker" if all_ok else "final_analysis_errors.marker"
    with open(os.path.join(LOGS, marker), "w") as f:
        f.write("done")


if __name__ == "__main__":
    main()
