#!/usr/bin/env python3
"""
Extract per-task token estimates from trajectory files.

Token counts are estimated from the trajectory text because the frameworks
discard the usage dict returned by get_chat(). The estimate uses the standard
approximation: 1 token ≈ 4 characters (works well for Gemini models).

At each step i the agent sends the full history up to that point, so we sum
the cumulative character lengths across all steps to approximate total input
tokens, then add a fixed estimate for output tokens (actions are short).

Outputs:
  alfworld_runs/memory_retrieval_v2/memory_agent_runs/summaries/token_counts_seen.csv
  alfworld_runs/memory_retrieval_v2/memory_agent_runs/summaries/token_counts_unseen.csv
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── constants ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_ROOT = BASE_DIR / "alfworld_runs" / "memory_retrieval_v2" / "memory_agent_runs"
SUMMARIES_DIR = RUNS_ROOT / "summaries"
PROMPTS_PATH = BASE_DIR / "data" / "alfworld" / "prompts" / "alfworld_3prompts.json"

CHARS_PER_TOKEN = 4  # standard approximation for Gemini

FRAMEWORK_ORDER = [
    "react",
    "react_reflexion",
    "react_cr",
    "react_tr",
    "react_cr_tr",
    "react_hard_neg_cr_tr",
]

FRAMEWORK_DISPLAY = {
    "react":                  "ReAct",
    "react_reflexion":        "ReAct + Reflexion",
    "react_cr":               "ReAct + CR",
    "react_tr":               "ReAct + TR",
    "react_cr_tr":            "ReAct + CR + TR",
    "react_hard_neg_cr_tr":   "ReAct + hard_neg CR + TR",
}

SPLIT_ALIAS = {
    "valid_seen":   "seen",
    "valid_unseen": "unseen",
}


def load_prompts() -> Dict[str, str]:
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def choose_prompt_key(task_id: str) -> str:
    name = task_id.split("-")[0]
    if "pick_cool"  in name: return "react_cool_0"
    if "pick_heat"  in name: return "react_heat_0"
    if "pick_clean" in name: return "react_clean_0"
    if "pick_two"   in name: return "react_puttwo_0"
    if "look_at"    in name: return "react_examine_0"
    return "react_put_0"


def estimate_tokens_for_task(
    steps: List[Dict],
    base_prompt_chars: int,
    memory_chars: int = 0,
) -> int:
    """
    Estimate total tokens consumed for one task trajectory.

    At each step i the prompt sent to the LLM is approximately:
        base_prompt + memory + step_0 + step_1 + ... + step_{i-1}

    We sum those cumulative lengths, add a small fixed cost for output
    tokens (each action response is ~10–20 tokens), then convert to tokens.
    """
    cumulative_text_chars = 0
    total_input_chars = 0
    output_chars = 0

    for step in steps:
        # Input at this step = base_prompt + memory + everything seen so far
        total_input_chars += base_prompt_chars + memory_chars + cumulative_text_chars

        action_text = step.get("action", "")
        obs_text    = step.get("observation", "")
        step_chars  = len(action_text) + len(obs_text)

        output_chars       += len(action_text)          # LLM produced the action
        cumulative_text_chars += step_chars

    total_chars = total_input_chars + output_chars
    return max(1, total_chars // CHARS_PER_TOKEN)


def process_trajectories(
    traj_path: Path,
    prompts: Dict[str, str],
) -> List[Dict]:
    """Return list of {task_id, trial_num, tokens, success} for every entry."""
    if not traj_path.exists():
        return []

    with open(traj_path) as f:
        trajectories = json.load(f)

    results = []
    for entry in trajectories:
        task_id   = entry.get("task_id", "")
        trial_num = int(entry.get("trial_num", 1))
        success   = bool(entry.get("success", False))
        steps     = entry.get("steps", [])

        prompt_key   = choose_prompt_key(task_id)
        base_prompt  = prompts.get(prompt_key, prompts.get("react_put_0", ""))
        base_chars   = len(base_prompt)

        tokens = estimate_tokens_for_task(steps, base_chars)
        results.append({
            "task_id":   task_id,
            "trial_num": trial_num,
            "tokens":    tokens,
            "success":   success,
        })

    return results


def aggregate_by_task(entries: List[Dict]) -> Dict[str, Dict]:
    """
    For frameworks with multiple trials (Reflexion), sum tokens across all
    trials for the same task so we get total tokens spent on that task.
    """
    by_task: Dict[str, Dict] = {}
    for e in entries:
        tid = e["task_id"]
        if tid not in by_task:
            by_task[tid] = {"task_id": tid, "tokens": 0, "success": e["success"]}
        by_task[tid]["tokens"] += e["tokens"]
        # success=True if any trial succeeded
        if e["success"]:
            by_task[tid]["success"] = True
    return by_task


def build_table(split: str) -> List[Dict]:
    """Build flat list of {task_id, framework, tokens, success} rows."""
    prompts = load_prompts()
    split_alias = SPLIT_ALIAS[split]
    rows = []

    for fw in FRAMEWORK_ORDER:
        fw_dir = RUNS_ROOT / split_alias / fw
        traj_path = fw_dir / "trajectories.json"

        entries = process_trajectories(traj_path, prompts)
        if not entries:
            print(f"  [warn] no trajectories for {split}/{fw}")
            continue

        by_task = aggregate_by_task(entries)

        for task_data in by_task.values():
            rows.append({
                "task_id":   task_data["task_id"],
                "framework": FRAMEWORK_DISPLAY[fw],
                "tokens":    task_data["tokens"],
                "success":   task_data["success"],
            })

    return rows


def write_table(rows: List[Dict], out_path: Path, split: str) -> None:
    """Write CSV and print a pretty console summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "framework", "tokens", "success"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*70}")
    print(f"  TOKEN COUNTS — {split.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    print(f"{'Task ID':<65} {'Framework':<28} {'Tokens':>10}")
    print(f"{'-'*65} {'-'*28} {'-'*10}")

    current_fw = None
    for r in rows:
        fw = r["framework"]
        if fw != current_fw:
            if current_fw is not None:
                print()
            current_fw = fw
        short_id = r["task_id"].split("/")[0][:62]
        print(f"{short_id:<65} {fw:<28} {r['tokens']:>10,}")

    print(f"\nSaved to: {out_path}")


def main() -> None:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    for split, alias in SPLIT_ALIAS.items():
        print(f"\nProcessing split: {split} ...")
        rows = build_table(split)

        if not rows:
            print(f"  No data found for {split}. Has the run finished?")
            continue

        out_path = SUMMARIES_DIR / f"token_counts_{alias}.csv"
        write_table(rows, out_path, split)


if __name__ == "__main__":
    main()
