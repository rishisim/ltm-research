#!/usr/bin/env python3
"""
Extend the per-split summary tables with two new columns:
  - Avg Tokens / Task    : mean tokens spent per unique task (all trials included)
  - Avg Tokens / Success : mean tokens spent per successfully completed task

Reads:
  summaries/token_counts_seen.csv
  summaries/token_counts_unseen.csv
  summaries/summary_valid_seen.json
  summaries/summary_valid_unseen.json

Writes (updated):
  summaries/summary_valid_seen_with_tokens.csv
  summaries/summary_valid_seen_with_tokens.md
  summaries/summary_valid_unseen_with_tokens.csv
  summaries/summary_valid_unseen_with_tokens.md
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
SUMMARIES_DIR = BASE_DIR / "alfworld_runs" / "memory_retrieval_v2" / "memory_agent_runs" / "summaries"

FRAMEWORK_DISPLAY = {
    "react":                "ReAct",
    "react_reflexion":      "ReAct + Reflexion",
    "react_cr":             "ReAct + CR",
    "react_tr":             "ReAct + TR",
    "react_cr_tr":          "ReAct + CR + TR",
    "react_hard_neg_cr_tr": "ReAct + hard_neg CR + TR",
}

SPLIT_ALIAS = {
    "valid_seen":   "seen",
    "valid_unseen": "unseen",
}


def load_token_counts(alias: str) -> Dict[str, Dict[str, int]]:
    """
    Load token_counts_{alias}.csv.

    Returns:
        {framework_display_name: {"total_tokens": int, "success_tokens": int, "task_count": int, "success_count": int}}
    """
    path = SUMMARIES_DIR / f"token_counts_{alias}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Token counts file not found: {path}\n"
            f"Run scripts/extract_token_counts.py first."
        )

    agg: Dict[str, Dict] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fw      = row["framework"]
            tokens  = int(row["tokens"])
            success = row["success"].strip().lower() in ("true", "1", "yes")

            if fw not in agg:
                agg[fw] = {
                    "total_tokens":   0,
                    "success_tokens": 0,
                    "task_count":     0,
                    "success_count":  0,
                }
            agg[fw]["total_tokens"]   += tokens
            agg[fw]["task_count"]     += 1
            if success:
                agg[fw]["success_tokens"] += tokens
                agg[fw]["success_count"]  += 1

    return agg


def load_summary_json(split: str) -> List[Dict]:
    path = SUMMARIES_DIR / f"summary_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["rows"]


def compute_avg_tokens(token_data: Dict) -> tuple:
    """Returns (avg_tokens_per_task, avg_tokens_per_success)."""
    n      = token_data["task_count"]
    n_succ = token_data["success_count"]
    total  = token_data["total_tokens"]
    succ_t = token_data["success_tokens"]

    avg_per_task    = round(total  / n,      0) if n      > 0 else 0
    avg_per_success = round(succ_t / n_succ, 0) if n_succ > 0 else 0
    return int(avg_per_task), int(avg_per_success)


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: List[Dict], path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# Summary with Token Counts ({split})\n\n")
        f.write(
            "| Framework | Success / Total | Accuracy | Avg Steps/Trial | "
            "Avg Steps/Task | Gain from ReAct | Avg Tokens/Task | Avg Tokens/Success |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['framework']} "
                f"| {r['success_total']} "
                f"| {float(r['accuracy']):.2%} "
                f"| {float(r['avg_steps_per_trial']):.2f} "
                f"| {float(r['avg_steps_per_task']):.2f} "
                f"| {float(r['gain_from_base_react']):+.2%} "
                f"| {r['avg_tokens_per_task']:,} "
                f"| {r['avg_tokens_per_success']:,} |\n"
            )


def print_table(rows: List[Dict], split: str) -> None:
    col_w = [28, 16, 10, 18, 16, 18, 18, 20]
    headers = [
        "Framework", "Success/Total", "Accuracy",
        "Avg Steps/Trial", "Avg Steps/Task", "Gain from ReAct",
        "Avg Tokens/Task", "Avg Tokens/Success",
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_w)) + "|"

    print(f"\n{'='*sum(col_w) + len(col_w)*3 + len(col_w)+1}")
    print(f"  SUMMARY WITH TOKEN COUNTS — {split.upper().replace('_', ' ')}")
    print(sep)
    print(header_row)
    print(sep)
    for r in rows:
        vals = [
            str(r["framework"])[:col_w[0]],
            str(r["success_total"]),
            f"{float(r['accuracy']):.2%}",
            f"{float(r['avg_steps_per_trial']):.2f}",
            f"{float(r['avg_steps_per_task']):.2f}",
            f"{float(r['gain_from_base_react']):+.2%}",
            f"{r['avg_tokens_per_task']:,}",
            f"{r['avg_tokens_per_success']:,}",
        ]
        print("|" + "|".join(f" {v:<{w}} " for v, w in zip(vals, col_w)) + "|")
    print(sep)


def process_split(split: str, alias: str) -> None:
    print(f"\nProcessing {split} ...")

    summary_rows = load_summary_json(split)
    token_data   = load_token_counts(alias)

    enriched = []
    for row in summary_rows:
        fw_display = row.get("framework", "")
        td = token_data.get(fw_display)

        if td:
            avg_per_task, avg_per_success = compute_avg_tokens(td)
        else:
            print(f"  [warn] no token data for framework: {fw_display!r}")
            avg_per_task = avg_per_success = 0

        enriched.append({
            "framework":            row["framework"],
            "success_total":        row["success_total"],
            "accuracy":             row["accuracy"],
            "avg_steps_per_trial":  row["avg_steps_per_trial"],
            "avg_steps_per_task":   row["avg_steps_per_task"],
            "gain_from_base_react": row["gain_from_base_react"],
            "avg_tokens_per_task":  avg_per_task,
            "avg_tokens_per_success": avg_per_success,
        })

    csv_out = SUMMARIES_DIR / f"summary_{split}_with_tokens.csv"
    md_out  = SUMMARIES_DIR / f"summary_{split}_with_tokens.md"

    write_csv(enriched, csv_out)
    write_md(enriched, md_out, split)
    print_table(enriched, split)

    print(f"\n  Saved: {csv_out.name}")
    print(f"  Saved: {md_out.name}")


def main() -> None:
    errors = []
    for split, alias in SPLIT_ALIAS.items():
        try:
            process_split(split, alias)
        except FileNotFoundError as e:
            errors.append(str(e))
            print(f"  [skip] {e}")

    if errors:
        print(f"\n{'='*60}")
        print("Some splits were skipped due to missing files.")
        print("Make sure both runs are complete and you have run:")
        print("  python scripts/extract_token_counts.py")


if __name__ == "__main__":
    main()
