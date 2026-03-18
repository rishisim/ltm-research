#!/usr/bin/env python3
"""
Print a formatted results table for WebShop framework suite runs.

Reads metrics.json from each framework directory and prints a
summary table matching the research report format. Safe to run while
a suite is still in progress — completed frameworks are shown, pending
ones are marked as such.

Usage:
    python scripts/print_webshop_results.py
    python scripts/print_webshop_results.py --runs-root webshop_runs/memory_retrieval_v2/memory_agent_runs
"""

import argparse
import json
from pathlib import Path

FRAMEWORK_ORDER = [
    "react",
    "react_reflexion",
    "react_cr",
    "react_tr",
    "react_cr_tr",
    "react_hard_neg_cr_tr",
]

FRAMEWORK_DISPLAY = {
    "react":                 "ReAct",
    "react_reflexion":       "ReAct + Reflexion",
    "react_cr":              "ReAct + CR",
    "react_tr":              "ReAct + TR",
    "react_cr_tr":           "ReAct + CR + TR",
    "react_hard_neg_cr_tr":  "ReAct + hard_neg CR + TR",
}

# WebShop runs use {framework}_{split} directory naming
# Only dev is used for evaluation; train was used for KB generation only.
SPLITS = {
    "dev": "Dev",
}


def load_metrics(runs_root: Path, framework_id: str, split: str):
    """Return metrics dict or None if the run hasn't completed yet."""
    dir_name = f"{framework_id}_{split}"
    path = runs_root / dir_name / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # WebShop metrics.json stores fields at top level (not nested under "metrics")
    if "metrics" in data:
        return data["metrics"]
    return data


def print_table(split_label: str, rows: list[dict]) -> None:
    col_w = [32, 16, 10, 22, 20, 22]
    headers = ["Framework", "Success / Total", "Accuracy",
               "Avg Steps/Trial", "Avg Steps/Task", "Gain from ReAct"]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f"{{:<{w}}}" for w in col_w) + "|"

    print(f"\n{'=' * (sum(col_w) + len(col_w) + 1)}")
    print(f"  WebShop — {split_label}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)

    for row in rows:
        if row.get("pending"):
            line = fmt.format(
                row["framework"], "(in progress)", "", "", "", ""
            )
        else:
            gain = row["gain"]
            gain_str = f"{gain:+.2%}" if gain != 0 else "—"
            line = fmt.format(
                row["framework"],
                row["success_total"],
                f"{row['accuracy']:.2%}",
                f"{row['avg_steps_per_trial']:.2f}",
                f"{row['avg_steps_per_task']:.2f}",
                gain_str,
            )
        print(line)

    print(sep)


def build_rows(runs_root: Path, split: str) -> list[dict]:
    rows = []
    base_accuracy = None

    for fw in FRAMEWORK_ORDER:
        m = load_metrics(runs_root, fw, split)
        if m is None:
            continue

        if base_accuracy is None and fw == "react":
            base_accuracy = m["accuracy"]

        gain = m["accuracy"] - (base_accuracy or 0.0)
        rows.append({
            "framework":           FRAMEWORK_DISPLAY[fw],
            "success_total":       m["success_total"],
            "accuracy":            m["accuracy"],
            "avg_steps_per_trial": m["avg_steps_per_trial"],
            "avg_steps_per_task":  m["avg_steps_per_task"],
            "gain":                gain,
            "pending":             False,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Print WebShop suite results")
    parser.add_argument(
        "--runs-root",
        default="webshop_runs/memory_retrieval_v2/memory_agent_runs",
        help="Path to the memory_agent_runs directory",
    )
    parser.add_argument(
        "--split",
        choices=list(SPLITS.keys()),
        default=None,
        help="Only show results for a specific split (default: all available)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    runs_root = (base_dir / args.runs_root).resolve()

    if not runs_root.exists():
        print(f"Runs root not found: {runs_root}")
        return

    splits_to_show = [(args.split, SPLITS[args.split])] if args.split else list(SPLITS.items())

    for split, label in splits_to_show:
        rows = build_rows(runs_root, split)
        if any(not r.get("pending") for r in rows):
            print_table(label, rows)
        else:
            print(f"\n[{label}] No results yet.")


if __name__ == "__main__":
    main()
