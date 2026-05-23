#!/usr/bin/env python3
"""Aggregate final LTM runs from raw attempts/trajectories.

This script is deliberately downstream of audit_final_run.py's loading and
metric helpers. It does not read summary CSVs, so paper tables can be regenerated
from the raw run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.analysis.audit_final_run import (  # noqa: E402
    JsonIssue,
    as_bool,
    as_float,
    count_help_steps,
    discover_run_dirs,
    final_record_map,
    final_records,
    load_json,
    load_run_data,
    recompute_metrics,
    repo_commit,
    task_id,
    trial_num,
    write_json,
)


NUMERIC_AGGREGATE_FIELDS = [
    "task_count",
    "success_count",
    "success_rate",
    "avg_reward",
    "avg_steps",
    "context_tasks",
    "retrieved_learnings_total",
    "avg_retrieved_learnings_per_task",
    "help_calls_total",
    "avg_help_calls_per_task",
    "total_steps",
    "avg_steps_from_trajectories",
    "input_tokens_total",
    "output_tokens_total",
    "total_tokens_total",
    "cached_tokens_total",
    "avg_total_tokens_per_task",
    "cost_total",
    "avg_cost_per_task",
]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_suite_roots(paths: Sequence[Path]) -> List[Path]:
    roots: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if (resolved / "suite_config.json").exists():
            roots.add(resolved)
            continue
        for config_path in resolved.rglob("suite_config.json"):
            if "archive" in config_path.parts or "misc" in config_path.parts:
                continue
            roots.add(config_path.parent)
    return sorted(roots)


def load_suite_config(root: Path) -> Dict[str, Any]:
    issues: List[JsonIssue] = []
    payload = load_json(root / "suite_config.json", issues)
    return payload if isinstance(payload, dict) else {}


def success_value(row: Dict[str, Any], reward_threshold: float) -> bool:
    reward = as_float(row.get("reward"))
    if reward is not None:
        return reward >= reward_threshold
    value = as_bool(row.get("success"))
    return bool(value)


def latest_trajectory_map(trajectories: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
    for index, row in enumerate(trajectories):
        tid = task_id(row)
        if not tid:
            continue
        candidate = (trial_num(row), index, row)
        if tid not in latest or candidate[:2] >= latest[tid][:2]:
            latest[tid] = candidate
    return {tid: value[2] for tid, value in latest.items()}


def sum_step_token_field(steps: Iterable[Dict[str, Any]], field: str) -> float:
    total = 0.0
    for step in steps:
        if not isinstance(step, dict):
            continue
        token_usage = step.get("token_usage")
        if not isinstance(token_usage, dict):
            continue
        value = as_float(token_usage.get(field))
        if value is not None:
            total += value
    return total


def record_token_total(row: Dict[str, Any], field: str) -> float:
    value = as_float(row.get(field))
    if value is not None:
        return value
    return sum_step_token_field(row.get("steps") or [], field)


def record_cost(row: Dict[str, Any]) -> float:
    for key in ("cost", "total_cost", "cost_usd", "total_cost_usd", "api_cost_usd"):
        value = as_float(row.get(key))
        if value is not None:
            return value
    step_total = 0.0
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        token_usage = step.get("token_usage")
        if isinstance(token_usage, dict):
            for key in ("cost", "cost_usd", "total_cost_usd"):
                value = as_float(token_usage.get(key))
                if value is not None:
                    step_total += value
                    break
    return step_total


def usage_summary(run: Any) -> Dict[str, Any]:
    rows = final_records(run)
    trajectories_by_task = latest_trajectory_map(run.trajectories)

    context_tasks = 0
    retrieved_total = 0
    help_total = 0
    total_steps = 0.0
    step_count_records = 0
    input_tokens = 0.0
    output_tokens = 0.0
    total_tokens = 0.0
    cached_tokens = 0.0
    cost_total = 0.0

    for row in rows:
        tid = task_id(row)
        source = trajectories_by_task.get(tid, row)
        context = source.get("context_from_retrieval")
        context_count = len(context) if isinstance(context, list) else 0
        if context_count:
            context_tasks += 1
            retrieved_total += context_count

        help_count = source.get("help_call_count")
        if help_count is None:
            help_count = count_help_steps(source)
        try:
            help_total += int(help_count or 0)
        except (TypeError, ValueError):
            help_total += count_help_steps(source)

        step_value = as_float(source.get("step_num"))
        if step_value is None and isinstance(source.get("steps"), list):
            step_value = float(len(source["steps"]))
        if step_value is not None:
            total_steps += step_value
            step_count_records += 1

        input_tokens += record_token_total(source, "input_tokens")
        output_tokens += record_token_total(source, "output_tokens")
        total_tokens += record_token_total(source, "total_tokens")
        cached_tokens += record_token_total(source, "cached_tokens")
        cost_total += record_cost(source)

    task_count = len(rows)
    return {
        "context_tasks": context_tasks,
        "retrieved_learnings_total": retrieved_total,
        "avg_retrieved_learnings_per_task": retrieved_total / task_count if task_count else 0.0,
        "help_calls_total": help_total,
        "avg_help_calls_per_task": help_total / task_count if task_count else 0.0,
        "total_steps": total_steps,
        "avg_steps_from_trajectories": total_steps / step_count_records if step_count_records else None,
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "total_tokens_total": total_tokens,
        "cached_tokens_total": cached_tokens,
        "avg_total_tokens_per_task": total_tokens / task_count if task_count else 0.0,
        "cost_total": cost_total,
        "avg_cost_per_task": cost_total / task_count if task_count else 0.0,
    }


def aggregate_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str], numeric_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)

    output: List[Dict[str, Any]] = []
    for key_values, group_rows in sorted(groups.items()):
        out = {key: value for key, value in zip(keys, key_values)}
        out["seed_count"] = len({str(row.get("seed", "")) for row in group_rows})
        for field in numeric_fields:
            values = [as_float(row.get(field)) for row in group_rows]
            values = [value for value in values if value is not None and math.isfinite(value)]
            out[f"{field}_mean"] = mean(values) if values else None
            out[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(out)
    return output


def paired_delta_rows(per_run_records: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]], reward_threshold: float) -> List[Dict[str, Any]]:
    by_group_split_seed: Dict[Tuple[str, str, str], Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    for (group, split, seed, framework), records in per_run_records.items():
        by_group_split_seed[(group, split, seed)][framework] = records

    rows: List[Dict[str, Any]] = []
    for (group, split, seed), framework_maps in sorted(by_group_split_seed.items()):
        react_map = framework_maps.get("react")
        if not react_map:
            continue
        for framework, record_map in sorted(framework_maps.items()):
            if framework == "react":
                continue
            common = sorted(set(react_map) & set(record_map))
            if not common:
                continue
            react_success = [success_value(react_map[tid], reward_threshold) for tid in common]
            other_success = [success_value(record_map[tid], reward_threshold) for tid in common]
            react_rewards = [as_float(react_map[tid].get("reward")) for tid in common]
            other_rewards = [as_float(record_map[tid].get("reward")) for tid in common]
            paired_rewards = [
                (base, other)
                for base, other in zip(react_rewards, other_rewards)
                if base is not None and other is not None
            ]
            improved = sum((not base) and other for base, other in zip(react_success, other_success))
            regressed = sum(base and (not other) for base, other in zip(react_success, other_success))
            rows.append({
                "run_group": group,
                "split": split,
                "seed": seed,
                "framework": framework,
                "baseline": "react",
                "common_tasks": len(common),
                "success_rate_react": sum(react_success) / len(common),
                "success_rate_framework": sum(other_success) / len(common),
                "success_rate_delta": (sum(other_success) - sum(react_success)) / len(common),
                "avg_reward_react": mean([pair[0] for pair in paired_rewards]) if paired_rewards else None,
                "avg_reward_framework": mean([pair[1] for pair in paired_rewards]) if paired_rewards else None,
                "avg_reward_delta": mean([pair[1] - pair[0] for pair in paired_rewards]) if paired_rewards else None,
                "improved_tasks": improved,
                "regressed_tasks": regressed,
                "unchanged_success_tasks": len(common) - improved - regressed,
            })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate audited final LTM runs")
    parser.add_argument(
        "--runs-root",
        action="append",
        type=Path,
        default=[],
        help="Run root or parent containing suite_config.json files. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_reports/final/aggregates"),
        help="Directory for aggregate JSON/CSV outputs",
    )
    parser.add_argument("--reward-threshold", type=float, default=None, help="Override success threshold")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = discover_suite_roots(args.runs_root or [Path("final_runs/eval")])
    if not roots:
        raise SystemExit("No suite roots found. Expected directories containing suite_config.json.")

    per_seed_rows: List[Dict[str, Any]] = []
    per_run_records: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = {}
    load_warnings: List[str] = []

    for root in roots:
        config = load_suite_config(root)
        threshold = args.reward_threshold if args.reward_threshold is not None else float(config.get("reward_threshold", 1.0) or 1.0)
        run_group = root.name
        run_dirs = discover_run_dirs(root)
        for run_dir in run_dirs:
            run = load_run_data(root, run_dir, config)
            if run.json_issues:
                load_warnings.extend(f"{issue.path}: {issue.detail}" for issue in run.json_issues)
            metrics = recompute_metrics(run, threshold)
            usage = usage_summary(run)
            row = {
                "run_group": run_group,
                "run_root": str(root),
                "split": run.split,
                "framework": run.framework,
                "seed": run.seed,
                "run_dir": str(run.run_dir),
                **metrics,
                **usage,
            }
            per_seed_rows.append(row)
            per_run_records[(run_group, run.split, run.seed, run.framework)] = final_record_map(run)

    framework_aggregate = aggregate_rows(
        per_seed_rows,
        keys=["run_group", "split", "framework"],
        numeric_fields=NUMERIC_AGGREGATE_FIELDS,
    )

    paired_rows = paired_delta_rows(per_run_records, args.reward_threshold or 1.0)
    paired_aggregate = aggregate_rows(
        paired_rows,
        keys=["run_group", "split", "framework", "baseline"],
        numeric_fields=[
            "common_tasks",
            "success_rate_react",
            "success_rate_framework",
            "success_rate_delta",
            "avg_reward_react",
            "avg_reward_framework",
            "avg_reward_delta",
            "improved_tasks",
            "regressed_tasks",
            "unchanged_success_tasks",
        ],
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "final_run_per_seed_metrics.csv", per_seed_rows)
    write_csv(output_dir / "final_run_framework_aggregate.csv", framework_aggregate)
    write_csv(output_dir / "final_run_paired_deltas_vs_react.csv", paired_rows)
    write_csv(output_dir / "final_run_paired_deltas_vs_react_aggregate.csv", paired_aggregate)
    write_csv(
        output_dir / "final_run_memory_usage.csv",
        per_seed_rows,
        fieldnames=[
            "run_group",
            "split",
            "framework",
            "seed",
            "task_count",
            "context_tasks",
            "retrieved_learnings_total",
            "avg_retrieved_learnings_per_task",
            "help_calls_total",
            "avg_help_calls_per_task",
            "run_dir",
        ],
    )
    write_csv(
        output_dir / "final_run_step_cost_token_summary.csv",
        per_seed_rows,
        fieldnames=[
            "run_group",
            "split",
            "framework",
            "seed",
            "task_count",
            "total_steps",
            "avg_steps_from_trajectories",
            "input_tokens_total",
            "output_tokens_total",
            "total_tokens_total",
            "cached_tokens_total",
            "avg_total_tokens_per_task",
            "cost_total",
            "avg_cost_per_task",
            "run_dir",
        ],
    )

    report = {
        "audit": {
            "script": str(Path(__file__).resolve()),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "git_commit": repo_commit(PROJECT_ROOT),
            "suite_roots": [str(root) for root in roots],
        },
        "counts": {
            "suite_roots": len(roots),
            "per_seed_rows": len(per_seed_rows),
            "framework_aggregate_rows": len(framework_aggregate),
            "paired_delta_rows": len(paired_rows),
        },
        "load_warnings": load_warnings,
        "outputs": {
            "per_seed_metrics": str(output_dir / "final_run_per_seed_metrics.csv"),
            "framework_aggregate": str(output_dir / "final_run_framework_aggregate.csv"),
            "paired_deltas": str(output_dir / "final_run_paired_deltas_vs_react.csv"),
            "paired_deltas_aggregate": str(output_dir / "final_run_paired_deltas_vs_react_aggregate.csv"),
            "memory_usage": str(output_dir / "final_run_memory_usage.csv"),
            "step_cost_token_summary": str(output_dir / "final_run_step_cost_token_summary.csv"),
        },
    }
    write_json(output_dir / "aggregate_final_runs_report.json", report)
    print(json.dumps(report["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
