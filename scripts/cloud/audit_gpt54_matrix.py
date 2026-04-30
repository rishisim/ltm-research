#!/usr/bin/env python3
"""Audit GPT-5.4 rerun outputs for completeness and WebShop action validity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


FRAMEWORKS = [
    "react",
    "react_cr",
    "react_tr",
    "react_cr_tr",
    "react_hard_neg_cr_tr",
]
SEEDS = [0, 1, 2]
VALID_WEBSHOP_PREFIXES = ("think:", "search[", "click[", "help[")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def metric_summary(path: Path) -> str:
    if not path.exists():
        return "MISSING metrics.json"
    data = load_json(path)
    metrics = data.get("metrics", data)
    success_total = metrics.get("success_total")
    accuracy = metrics.get("accuracy")
    avg_reward = metrics.get("avg_reward")
    attempts = metrics.get("attempt_count")
    return (
        f"{success_total or 'n/a'}"
        f" acc={accuracy if accuracy is not None else 'n/a'}"
        f" avg_reward={avg_reward if avg_reward is not None else 'n/a'}"
        f" attempts={attempts if attempts is not None else 'n/a'}"
    )


def trajectory_count(run_dir: Path) -> int:
    trajectories = run_dir / "trajectories.json"
    if trajectories.exists():
        return len(load_json(trajectories))
    return count_jsonl(run_dir / "agent_trajectories.jsonl")


def audit_matrix(root: Path, split: str, expected: int, label: str) -> list[str]:
    rows: list[str] = []
    for framework in FRAMEWORKS:
        for seed in SEEDS:
            run_dir = root / split / framework / f"seed_{seed}"
            n = trajectory_count(run_dir)
            status = "OK" if n == expected else f"GAP({n}/{expected})"
            rows.append(
                f"{label} {framework} seed_{seed}: {status} "
                f"{metric_summary(run_dir / 'metrics.json')}"
            )
    return rows


def iter_webshop_actions(path: Path) -> Iterable[str]:
    if not path.exists():
        return []
    trajectories = load_json(path)
    actions: list[str] = []
    for traj in trajectories:
        for step in traj.get("steps", []):
            if step.get("label") == "action":
                actions.append((step.get("value") or "").strip())
    return actions


def audit_webshop_actions(root: Path, min_valid_rate: float) -> list[str]:
    rows: list[str] = []
    for framework in FRAMEWORKS:
        for seed in SEEDS:
            run_dir = root / "test" / framework / f"seed_{seed}"
            actions = list(iter_webshop_actions(run_dir / "trajectories.json"))
            counts = Counter()
            examples: list[str] = []
            for action in actions:
                if "Obs:" in action or "Action:" in action:
                    counts["leak"] += 1
                    examples.append(action[:120])
                elif action.startswith(VALID_WEBSHOP_PREFIXES):
                    if action.startswith(("search[", "click[", "help[")) and not action.endswith("]"):
                        counts["malformed"] += 1
                        examples.append(action[:120])
                    else:
                        counts["valid"] += 1
                elif action.startswith("Obs:"):
                    counts["obs_leak"] += 1
                    examples.append(action[:120])
                elif not action:
                    counts["empty"] += 1
                    examples.append("<empty>")
                else:
                    counts["invalid"] += 1
                    examples.append(action[:120])
            total = sum(counts.values())
            valid_rate = counts["valid"] / total if total else 0.0
            status = "OK" if total and valid_rate >= min_valid_rate else "CHECK"
            rows.append(
                f"WebShop actions {framework} seed_{seed}: {status} "
                f"valid={counts['valid']}/{total} rate={valid_rate:.3f} "
                f"other={dict(counts - Counter(valid=counts['valid']))}"
            )
            if examples:
                rows.append(f"  examples={examples[:3]}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--web-root", default=None)
    parser.add_argument("--alf-root", default=None)
    parser.add_argument("--sql-root", default=None)
    parser.add_argument(
        "--min-webshop-valid-rate",
        type=float,
        default=0.90,
        help=(
            "Minimum strict WebShop action-validity rate before reporting CHECK. "
            "The original WebShop runner has historical memory-variant runs around "
            "0.94-0.96 strict validity, so the default is intentionally not 0.98."
        ),
    )
    args = parser.parse_args()

    web_root = Path(args.web_root or f"webshop_runs/rerun_clean/{args.model}")
    alf_root = Path(args.alf_root or f"alfworld_runs/rerun_clean/{args.model}")
    sql_root = Path(args.sql_root or f"intercode_sql_runs/rerun_clean/{args.model}")

    print("=== Completeness ===")
    for row in audit_matrix(web_root, "test", 200, "WebShop"):
        print(row)
    for row in audit_matrix(alf_root, "unseen", 134, "ALFWorld"):
        print(row)
    for row in audit_matrix(sql_root, "test", 200, "SQL"):
        print(row)

    print("\n=== WebShop Action Validity ===")
    for row in audit_webshop_actions(web_root, args.min_webshop_valid_rate):
        print(row)


if __name__ == "__main__":
    main()
