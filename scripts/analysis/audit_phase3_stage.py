#!/usr/bin/env python3
"""Gate audit for staged Phase 3 final evaluations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FRAMEWORKS = ["react", "react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"]
MEMORY_FRAMEWORKS = {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}
CONTEXT_EXPECTED_FRAMEWORKS = {"react_cr", "react_cr_tr", "react_hard_neg_cr_tr"}
SEEDS = ["0", "1", "2"]

SCIENCEWORLD_TASK_IDS = [
    "boil",
    "change-the-state-of-matter-of",
    "chemistry-mix",
    "chemistry-mix-paint-secondary-color",
    "chemistry-mix-paint-tertiary-color",
    "find-animal",
    "find-living-thing",
    "find-non-living-thing",
    "find-plant",
    "freeze",
    "grow-fruit",
    "grow-plant",
    "identify-life-stages-1",
    "identify-life-stages-2",
    "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces",
    "inclined-plane-friction-unnamed-surfaces",
    "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "lifespan-shortest-lived",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
    "melt",
    "mendelian-genetics-known-plant",
    "mendelian-genetics-unknown-plant",
    "power-component",
    "power-component-renewable-vs-nonrenewable-energy",
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
    "use-thermometer",
]

STAGES = {
    "A": {"alfworld": 34, "sql": 50, "scienceworld": 8},
    "B": {"alfworld": 67, "sql": 100, "scienceworld": 15},
    "C": {"alfworld": 134, "sql": 200, "scienceworld": 30},
}

ENVIRONMENTS = {
    "alfworld": {
        "root": Path("final_runs/eval/alfworld_trials7_gemini"),
        "split_dir": "unseen",
        "config_split": "valid_unseen",
        "kb": Path("final_runs/kb/alfworld_train_reflexion_trials7/knowledge_base.json"),
        "count_key": "alfworld",
    },
    "sql": {
        "root": Path("final_runs/eval/sql_trials7_gemini"),
        "split_dir": "test",
        "config_split": "test",
        "kb": Path("final_runs/kb/sql_train_reflexion_trials7_sanitized/knowledge_base.sql_sanitized.json"),
        "count_key": "sql",
    },
    "scienceworld": {
        "root": Path("final_runs/eval/scienceworld_trials7_gemini"),
        "split_dir": "test",
        "config_split": "test",
        "kb": Path("final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat/knowledge_base.json"),
        "count_key": "scienceworld",
    },
}

VALID_LEVEL_PRIORITY = {"": 0, "CANDIDATE": 1, "VALID_SAME_TRIAL": 2, "VALID_NEXT_TRIAL": 3}
LLM_ERROR_MARKERS = ("LLM Error", "RetryError", "Payment Required", "402 Client Error")


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_json_optional(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def task_id(row: Dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("task_id_str") or "")


def suspected_llm_failure(row: Dict[str, Any]) -> bool:
    """Detect provider-failure records that still produced a final task row."""
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        return False

    total_tokens = 0
    empty_model_actions = 0
    empty_action_invalid_observation = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        observation = str(step.get("observation") or "")
        if any(marker in action or marker in observation for marker in LLM_ERROR_MARKERS):
            return True
        usage = step.get("token_usage") if isinstance(step.get("token_usage"), dict) else {}
        total_tokens += int(as_float(usage.get("total_tokens")) or 0)
        if not action.strip() and int(as_float(usage.get("total_tokens")) or 0) == 0:
            empty_model_actions += 1
            if "No known action matches that input" in observation:
                empty_action_invalid_observation += 1

    row_tokens = int(as_float(row.get("total_tokens")) or 0)
    one_empty_no_token_step = (
        len(steps) == 1
        and empty_model_actions == 1
        and empty_action_invalid_observation == 1
        and total_tokens == 0
        and row_tokens == 0
    )
    all_empty_no_token_steps = (
        empty_model_actions == len(steps)
        and empty_action_invalid_observation > 0
        and total_tokens == 0
        and row_tokens == 0
    )
    return one_empty_no_token_step or all_empty_no_token_steps


def run_rows(run_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    for name in ("attempts.json", "agent_trajectories.jsonl", "trajectories.jsonl", "trajectories.json"):
        path = run_dir / name
        if not path.exists():
            continue
        if name.endswith(".jsonl"):
            return load_jsonl(path), name
        payload = load_json_optional(path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)], name
    return [], ""


def primary_trajectory_rows(run_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    for name in ("agent_trajectories.jsonl", "trajectories.jsonl", "trajectories.json"):
        path = run_dir / name
        if not path.exists():
            continue
        if name.endswith(".jsonl"):
            return load_jsonl(path), name
        payload = load_json_optional(path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)], name
    return [], ""


def latest_by_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        tid = task_id(row)
        if not tid:
            continue
        try:
            trial = int(row.get("trial_num") or 0)
        except (TypeError, ValueError):
            trial = 0
        candidate = (trial, index, row)
        if tid not in latest or candidate[:2] >= latest[tid][:2]:
            latest[tid] = candidate
    return {tid: item[2] for tid, item in latest.items()}


def count_help_steps(row: Dict[str, Any]) -> int:
    count = 0
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip().lower()
        if step.get("is_help_call") is True or action.startswith("help[") or action.startswith("help ["):
            count += 1
    return count


def count_help_calls(row: Dict[str, Any]) -> int:
    logged_count = row.get("help_call_count")
    if logged_count is not None:
        try:
            return int(logged_count)
        except (TypeError, ValueError):
            pass

    help_calls = row.get("help_calls")
    if isinstance(help_calls, list):
        return len(help_calls)
    return count_help_steps(row)


def retrieval_records(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    json_path = run_dir / "knowledge_retrieval_bases.json"
    payload = load_json_optional(json_path)
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                value = dict(value)
                value.setdefault("task_id", key)
                rows.append(value)
    elif isinstance(payload, list):
        rows.extend(row for row in payload if isinstance(row, dict))

    for item in load_jsonl(run_dir / "knowledge_retrieval_bases.jsonl"):
        if "knowledge_retrieval_base" in item:
            rows.append(item)
        else:
            for key, value in item.items():
                if isinstance(value, dict):
                    value = dict(value)
                    value.setdefault("task_id", key)
                    rows.append(value)
    return rows


def selected_rows(record: Dict[str, Any], max_learnings: int) -> List[Dict[str, Any]]:
    selected = record.get("selected_learnings")
    if isinstance(selected, list):
        return [row for row in selected if isinstance(row, dict)]
    base = record.get("knowledge_retrieval_base")
    if not isinstance(base, list):
        return []
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    count = metadata.get("actual_selected_count")
    if not isinstance(count, int):
        count = metadata.get("pick_learning_count")
    if not isinstance(count, int):
        count = max_learnings
    return [row for row in base[:count] if isinstance(row, dict)]


def valid_level_ok(level: str, minimum: str) -> bool:
    return VALID_LEVEL_PRIORITY.get(level or "CANDIDATE", 0) >= VALID_LEVEL_PRIORITY[minimum]


def sql_288_recovered_code_fence_seeds(root: Path, split_dir: str) -> List[str]:
    recovered_seeds: List[str] = []
    for seed in SEEDS:
        run_dir = root / split_dir / "react_cr" / f"seed_{seed}"
        rows, _ = primary_trajectory_rows(run_dir)
        row = latest_by_task(rows).get("sql_288")
        if not row:
            continue
        reward = as_float(row.get("reward"))
        succeeded = bool(row.get("success")) or (reward is not None and reward >= 1.0)
        if not succeeded:
            continue
        for step in row.get("steps") or []:
            if isinstance(step, dict) and "```" in str(step.get("action") or ""):
                recovered_seeds.append(seed)
                break
    return recovered_seeds


def config_errors(env_name: str, cfg: Dict[str, Any], spec: Dict[str, Any], stage_target: int) -> List[str]:
    errors: List[str] = []
    checks = {
        "model": "gemini-2.5-flash",
        "embedding_provider": "gemini",
        "max_learnings": 5,
        "min_valid_level": "VALID_NEXT_TRIAL",
    }
    for key, expected in checks.items():
        if cfg.get(key) != expected:
            errors.append(f"{env_name} suite_config {key}={cfg.get(key)!r}, expected {expected!r}")
    if cfg.get("num_tasks") != stage_target:
        errors.append(f"{env_name} suite_config num_tasks={cfg.get('num_tasks')!r}, expected {stage_target}")
    splits = cfg.get("splits")
    if splits != [spec["config_split"]]:
        errors.append(f"{env_name} suite_config splits={splits!r}, expected {[spec['config_split']]!r}")
    frameworks = cfg.get("frameworks")
    if frameworks != FRAMEWORKS:
        errors.append(f"{env_name} suite_config frameworks={frameworks!r}, expected {FRAMEWORKS!r}")
    memory_bank = cfg.get("memory_bank")
    expected_kb = spec["kb"].resolve()
    if memory_bank is None or Path(str(memory_bank)).resolve() != expected_kb:
        errors.append(f"{env_name} suite_config memory_bank={memory_bank!r}, expected {str(expected_kb)!r}")
    if env_name == "scienceworld":
        expected_tasks = SCIENCEWORLD_TASK_IDS[:stage_target]
        if cfg.get("task_ids") != expected_tasks:
            errors.append("scienceworld suite_config task_ids do not match the stage prefix")
        if cfg.get("max_valid_actions") != 80:
            errors.append(f"scienceworld suite_config max_valid_actions={cfg.get('max_valid_actions')!r}, expected 80")
    return errors


def audit_run(
    env_name: str,
    root: Path,
    split_dir: str,
    framework: str,
    seed: str,
    expected_count: int,
    min_valid_level: str,
    max_learnings: int,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    run_dir = root / split_dir / framework / f"seed_{seed}"
    errors: List[str] = []
    warnings: List[str] = []
    rows, source = run_rows(run_dir)
    ids = [task_id(row) for row in rows if task_id(row)]
    counts = Counter(ids)
    duplicate_ids = sorted(tid for tid, count in counts.items() if count > 1)
    rewards = [as_float(row.get("reward")) for row in rows]
    reward_values = [value for value in rewards if value is not None]

    if not run_dir.exists():
        errors.append(f"missing run directory {run_dir}")
    if not source:
        errors.append(f"missing final records for {run_dir}")
    if len(counts) != expected_count:
        errors.append(f"{env_name}/{framework}/seed_{seed} has {len(counts)} unique tasks, expected {expected_count}")
    if duplicate_ids:
        errors.append(f"{env_name}/{framework}/seed_{seed} has duplicate final task IDs: {duplicate_ids[:10]}")

    context_total = 0
    help_total = 0
    retrieval_record_count = 0
    retrieval_selected_total = 0
    invalid_retrieval_levels = 0
    suspected_llm_failure_tasks: List[str] = []

    usage_rows, usage_source = primary_trajectory_rows(run_dir)
    usage_by_task = latest_by_task(usage_rows)

    diagnostic_rows = usage_rows or rows
    for row in diagnostic_rows:
        if suspected_llm_failure(row):
            suspected_llm_failure_tasks.append(task_id(row) or "<unknown>")
    suspected_llm_failure_tasks = sorted(set(suspected_llm_failure_tasks))

    for row in rows:
        usage_row = usage_by_task.get(task_id(row), row)
        context = row.get("context_from_retrieval")
        if not isinstance(context, list):
            context = usage_row.get("context_from_retrieval")
        if isinstance(context, list):
            context_total += len(context)
            for memory in context:
                if isinstance(memory, dict):
                    level = str(memory.get("valid_level") or "CANDIDATE")
                    if not valid_level_ok(level, min_valid_level):
                        invalid_retrieval_levels += 1
        help_total += count_help_calls(usage_row)

    records = retrieval_records(run_dir)
    retrieval_record_count = len(records)
    for record in records:
        selected = selected_rows(record, max_learnings)
        retrieval_selected_total += len(selected)
        for memory in selected:
            level = str(memory.get("valid_level") or "CANDIDATE")
            if not valid_level_ok(level, min_valid_level):
                invalid_retrieval_levels += 1

    if framework in CONTEXT_EXPECTED_FRAMEWORKS and retrieval_selected_total == 0 and context_total == 0:
        errors.append(f"{env_name}/{framework}/seed_{seed} did not retrieve any context memories")
    if framework == "react_tr" and help_total == 0:
        warnings.append(f"{env_name}/react_tr/seed_{seed} made zero help calls")
    if invalid_retrieval_levels:
        errors.append(
            f"{env_name}/{framework}/seed_{seed} retrieved {invalid_retrieval_levels} memories below {min_valid_level}"
        )
    if suspected_llm_failure_tasks:
        sample = suspected_llm_failure_tasks[:10]
        errors.append(
            f"{env_name}/{framework}/seed_{seed} has {len(suspected_llm_failure_tasks)} suspected LLM/provider failure records: {sample}"
        )

    successes = 0
    for row in rows:
        reward = as_float(row.get("reward"))
        if reward is not None:
            successes += int(reward >= 1.0)
        else:
            successes += int(bool(row.get("success")))

    summary = {
        "run_dir": str(run_dir),
        "record_source": source,
        "usage_record_source": usage_source or source,
        "unique_task_count": len(counts),
        "record_count": len(rows),
        "duplicate_task_ids": duplicate_ids,
        "success_count": successes,
        "avg_reward": mean(reward_values) if reward_values else None,
        "context_memory_count": context_total,
        "help_call_count": help_total,
        "retrieval_record_count": retrieval_record_count,
        "retrieval_selected_count": retrieval_selected_total,
        "suspected_llm_failure_count": len(suspected_llm_failure_tasks),
        "suspected_llm_failure_task_ids": suspected_llm_failure_tasks,
    }
    return summary, errors, warnings


def audit_stage(stage: str, repo_root: Path) -> Dict[str, Any]:
    targets = STAGES[stage]
    report: Dict[str, Any] = {
        "stage": stage,
        "targets": targets,
        "passed": True,
        "errors": [],
        "warnings": [],
        "notes": [],
        "environments": {},
    }

    for env_name, spec in ENVIRONMENTS.items():
        root = repo_root / spec["root"]
        expected_count = int(targets[spec["count_key"]])
        env_report: Dict[str, Any] = {
            "root": str(root),
            "expected_count": expected_count,
            "config": {},
            "runs": {},
        }
        cfg = load_json_optional(root / "suite_config.json")
        if not isinstance(cfg, dict):
            report["errors"].append(f"{env_name} missing or corrupt suite_config.json")
            cfg = {}
        env_report["config"] = cfg
        report["errors"].extend(config_errors(env_name, cfg, {**spec, "kb": repo_root / spec["kb"]}, expected_count))

        for framework in FRAMEWORKS:
            for seed in SEEDS:
                summary, errors, warnings = audit_run(
                    env_name,
                    root,
                    str(spec["split_dir"]),
                    framework,
                    seed,
                    expected_count,
                    "VALID_NEXT_TRIAL",
                    5,
                )
                env_report["runs"][f"{framework}/seed_{seed}"] = summary
                report["errors"].extend(errors)
                report["warnings"].extend(warnings)
        report["environments"][env_name] = env_report

    sql_spec = ENVIRONMENTS["sql"]
    sql_code_fence_seeds = sql_288_recovered_code_fence_seeds(
        repo_root / sql_spec["root"],
        str(sql_spec["split_dir"]),
    )
    if stage == "C" and sql_code_fence_seeds:
        seed_text = ", ".join(sql_code_fence_seeds)
        report["notes"].append(
            f"SQL react_cr task sql_288 emitted recovered bare code-fence actions in seeds {seed_text}; "
            "the environment rejected them as invalid SQL and the final task records still succeeded, "
            "so this is tracked as non-fatal model-formatting noise rather than metric corruption."
        )
    report["passed"] = not report["errors"]
    return report


def format_md(report: Dict[str, Any]) -> str:
    lines = [
        f"# Phase 3 Stage {report['stage']} Report",
        "",
        f"Gate status: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "This stage is a corruption/trend gate only, not final science.",
        "",
        "## Targets",
        "",
        "| Environment | Expected completed tasks per framework/seed |",
        "| --- | ---: |",
        f"| ALFWorld valid_unseen | {report['targets']['alfworld']} |",
        f"| SQL test | {report['targets']['sql']} |",
        f"| ScienceWorld test categories | {report['targets']['scienceworld']} |",
        "",
        "## Gate Checks",
        "",
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
    ]
    if report["errors"]:
        lines.append("### Errors")
        lines.append("")
        for item in report["errors"][:50]:
            lines.append(f"- {item}")
        if len(report["errors"]) > 50:
            lines.append(f"- ... {len(report['errors']) - 50} additional errors")
        lines.append("")
    if report["warnings"]:
        lines.append("### Warnings")
        lines.append("")
        for item in report["warnings"][:50]:
            lines.append(f"- {item}")
        if len(report["warnings"]) > 50:
            lines.append(f"- ... {len(report['warnings']) - 50} additional warnings")
        lines.append("")
    if report.get("notes"):
        lines.append("### Notes")
        lines.append("")
        for item in report["notes"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend([
        "## Run Summary",
        "",
        "| Environment | Framework | Seed | Tasks | Successes | Avg Reward | Context Memories | Help Calls | Retrieval Records |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for env_name, env_report in report["environments"].items():
        for key, summary in env_report["runs"].items():
            framework, seed = key.split("/seed_")
            avg_reward = summary["avg_reward"]
            avg = "" if avg_reward is None else f"{avg_reward:.4f}"
            lines.append(
                f"| {env_name} | {framework} | {seed} | {summary['unique_task_count']} | "
                f"{summary['success_count']} | {avg} | {summary['context_memory_count']} | "
                f"{summary['help_call_count']} | {summary['retrieval_record_count']} |"
            )
    lines.extend([
        "",
        "## Continuation Decision",
        "",
        (
            "Continue to the next stage only after this report passes all hard gates."
            if not report["passed"]
            else "The stage passed the hard gates and is safe to continue to the next stage."
        ),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit staged Phase 3 final eval outputs")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    report = audit_stage(args.stage, repo_root)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(format_md(report).rstrip() + "\n")
    print(json.dumps({"stage": args.stage, "passed": report["passed"], "errors": len(report["errors"]), "warnings": len(report["warnings"])}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
