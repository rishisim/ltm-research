#!/usr/bin/env python3
"""Audit final LTM evaluation runs from raw artifacts.

The final campaign uses this script as the gate between run generation and
paper-facing aggregation. It intentionally recomputes task metrics from
attempt/trajectory records rather than trusting summary CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VALID_LEVEL_PRIORITY = {
    "": 0,
    "CANDIDATE": 1,
    "VALID_SAME_TRIAL": 2,
    "VALID_NEXT_TRIAL": 3,
}

CONTEXT_FRAMEWORKS = {"react_cr", "react_cr_tr", "react_hard_neg_cr_tr"}
MEMORY_FRAMEWORKS = {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}
REWARD_EPS = 1e-9
HELP_VALIDATION_RE = re.compile(r"Validation:\s*([A-Z_]+)")
LABEL_LEAKAGE_RE = re.compile(r"(?im)(?:^|\n)\s*(?:Obs|Observation|Action)\s*:")
NEXT_ACTION_RE = re.compile(
    r"(?im)\n\s*(?:think:|help\s*\[|search\s*\[|click\s*\[|buy\s*\[|submit\b)"
)
SQL_START_RE = re.compile(
    r"^\s*(?:SELECT|WITH|SHOW|DESCRIBE|DESC|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|PRAGMA|EXPLAIN)\b",
    re.IGNORECASE,
)


@dataclass
class JsonIssue:
    path: str
    detail: str


@dataclass
class RunData:
    run_root: Path
    run_dir: Path
    split: str
    framework: str
    seed: str
    suite_config: Dict[str, Any]
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_files: List[str] = field(default_factory=list)
    json_issues: List[JsonIssue] = field(default_factory=list)

    @property
    def rel_run_dir(self) -> str:
        try:
            return str(self.run_dir.relative_to(self.run_root))
        except ValueError:
            return str(self.run_dir)


def load_json(path: Path, issues: List[JsonIssue]) -> Any:
    try:
        with path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        issues.append(JsonIssue(str(path), f"JSON decode error at line {exc.lineno}: {exc.msg}"))
    except OSError as exc:
        issues.append(JsonIssue(str(path), f"Could not read file: {exc}"))
    return None


def load_jsonl(path: Path, issues: List[JsonIssue]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(JsonIssue(str(path), f"JSONL decode error at line {line_no}: {exc.msg}"))
                    continue
                if isinstance(value, dict):
                    rows.append(value)
                else:
                    issues.append(JsonIssue(str(path), f"Line {line_no} is {type(value).__name__}, expected object"))
    except OSError as exc:
        issues.append(JsonIssue(str(path), f"Could not read file: {exc}"))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def repo_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def parse_expected_counts(values: Sequence[str]) -> Tuple[Dict[str, int], Optional[int]]:
    split_counts: Dict[str, int] = {}
    default_count: Optional[int] = None
    for value in values:
        if "=" in value:
            split, count = value.split("=", 1)
            split_counts[split.strip()] = int(count)
        else:
            default_count = int(value)
    return split_counts, default_count


def split_aliases(split: str) -> set[str]:
    aliases = {split}
    if split == "seen":
        aliases.add("valid_seen")
    if split == "unseen":
        aliases.add("valid_unseen")
    if split == "valid_seen":
        aliases.add("seen")
    if split == "valid_unseen":
        aliases.add("unseen")
    return aliases


def expected_count_from_manifests(run_root: Path, split: str) -> Optional[int]:
    manifest_dir = run_root / "manifests"
    if not manifest_dir.exists():
        return None
    aliases = split_aliases(split)
    for path in sorted(manifest_dir.glob("*.json")):
        issues: List[JsonIssue] = []
        payload = load_json(path, issues)
        if issues or not isinstance(payload, dict):
            continue
        manifest_split = str(payload.get("split", ""))
        if manifest_split in aliases:
            for key in ("num_tasks_selected", "task_count", "num_tasks"):
                value = payload.get(key)
                if isinstance(value, int):
                    return value
            tasks = payload.get("tasks")
            if isinstance(tasks, list):
                return len(tasks)
    return None


def expected_task_count(
    run: RunData,
    split_counts: Dict[str, int],
    default_count: Optional[int],
) -> Optional[int]:
    for alias in split_aliases(run.split):
        if alias in split_counts:
            return split_counts[alias]
    if default_count is not None:
        return default_count
    manifest_count = expected_count_from_manifests(run.run_root, run.split)
    if manifest_count is not None:
        return manifest_count
    value = run.suite_config.get("num_tasks")
    return int(value) if isinstance(value, int) else None


def discover_run_dirs(run_root: Path) -> List[Path]:
    candidates: set[Path] = set()
    ignored_parts = {"archive", "diagnostics", "manifests", "misc", "summaries"}
    marker_names = {
        "metrics.json",
        "attempts.json",
        "trajectories.json",
        "trajectories.jsonl",
        "agent_trajectories.jsonl",
    }
    for marker in marker_names:
        for path in run_root.rglob(marker):
            if ignored_parts & set(path.parts):
                continue
            candidates.add(path.parent)
    return sorted(candidates)


def parse_run_identity(run_root: Path, run_dir: Path, suite_config: Dict[str, Any]) -> Tuple[str, str, str]:
    try:
        parts = run_dir.relative_to(run_root).parts
    except ValueError:
        parts = run_dir.parts

    seed = ""
    framework = run_dir.name
    split = ""

    if len(parts) >= 3 and parts[-1].startswith("seed_"):
        seed = parts[-1].split("seed_", 1)[1]
        framework = parts[-2]
        split = parts[-3]
    elif len(parts) >= 2:
        framework = parts[-1]
        split = parts[-2]

    if not seed:
        seed_value = suite_config.get("seed")
        seed = str(seed_value) if seed_value is not None else "0"
    return split, framework, seed


def normalize_retrieval_records(payload: Any, source_path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                task_id = str(item.get("task_id") or item.get("query") or index)
                records[task_id] = item
        return records

    if not isinstance(payload, dict):
        return records

    if "knowledge_retrieval_base" in payload or "selected_learnings" in payload:
        task_id = str(payload.get("task_id") or payload.get("query") or source_path.stem)
        records[task_id] = payload
        return records

    for key, value in payload.items():
        if isinstance(value, dict):
            value.setdefault("task_id", str(key))
            records[str(key)] = value
    return records


def load_retrieval_records(run_dir: Path, issues: List[JsonIssue]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    json_path = run_dir / "knowledge_retrieval_bases.json"
    jsonl_path = run_dir / "knowledge_retrieval_bases.jsonl"
    if json_path.exists():
        payload = load_json(json_path, issues)
        records.update(normalize_retrieval_records(payload, json_path))
    if jsonl_path.exists():
        for line_payload in load_jsonl(jsonl_path, issues):
            records.update(normalize_retrieval_records(line_payload, jsonl_path))
    return records


def load_run_data(run_root: Path, run_dir: Path, suite_config: Dict[str, Any]) -> RunData:
    split, framework, seed = parse_run_identity(run_root, run_dir, suite_config)
    run = RunData(run_root=run_root, run_dir=run_dir, split=split, framework=framework, seed=seed, suite_config=suite_config)

    attempts_path = run_dir / "attempts.json"
    if attempts_path.exists():
        payload = load_json(attempts_path, run.json_issues)
        if isinstance(payload, list):
            run.attempts = [row for row in payload if isinstance(row, dict)]
        elif payload is not None:
            run.json_issues.append(JsonIssue(str(attempts_path), "Expected attempts.json to contain a list"))

    trajectory_sources = [
        run_dir / "agent_trajectories.jsonl",
        run_dir / "trajectories.jsonl",
        run_dir / "trajectories.json",
    ]
    for path in trajectory_sources:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            run.trajectories = load_jsonl(path, run.json_issues)
        else:
            payload = load_json(path, run.json_issues)
            if isinstance(payload, list):
                run.trajectories = [row for row in payload if isinstance(row, dict)]
            elif payload is not None:
                run.json_issues.append(JsonIssue(str(path), "Expected trajectories JSON to contain a list"))
        break

    if not run.attempts and not run.trajectories:
        run.missing_files.append("attempts.json or trajectories/agent_trajectories file")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        run.missing_files.append("metrics.json")
    else:
        load_json(metrics_path, run.json_issues)

    run.retrieval_records = load_retrieval_records(run_dir, run.json_issues)
    if framework in CONTEXT_FRAMEWORKS and not run.retrieval_records:
        run.missing_files.append("knowledge_retrieval_bases.json/jsonl")

    return run


def task_id(row: Dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("task_id_str") or "")


def trial_num(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("trial_num", 0) or 0)
    except (TypeError, ValueError):
        return 0


def final_records(run: RunData) -> List[Dict[str, Any]]:
    if run.attempts:
        return run.attempts

    by_task: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
    for index, row in enumerate(run.trajectories):
        tid = task_id(row)
        if not tid:
            continue
        candidate = (trial_num(row), index, row)
        if tid not in by_task or candidate[:2] >= by_task[tid][:2]:
            by_task[tid] = candidate
    return [value[2] for value in sorted(by_task.values(), key=lambda item: item[1])]


def final_record_map(run: RunData) -> Dict[str, Dict[str, Any]]:
    return {task_id(row): row for row in final_records(run) if task_id(row)}


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "success", "1", "yes"}:
            return True
        if lowered in {"false", "fail", "failure", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def recompute_metrics(run: RunData, reward_threshold: float) -> Dict[str, Any]:
    rows = final_records(run)
    rewards = [as_float(row.get("reward")) for row in rows]
    reward_values = [value for value in rewards if value is not None]
    successes: List[bool] = []
    for row in rows:
        reward = as_float(row.get("reward"))
        if reward is not None:
            successes.append(reward >= reward_threshold)
        else:
            successes.append(bool(row.get("success")))
    step_values = [as_float(row.get("step_num")) for row in rows]
    step_values = [value for value in step_values if value is not None]
    return {
        "task_count": len(rows),
        "success_count": int(sum(successes)),
        "success_rate": float(sum(successes) / len(rows)) if rows else 0.0,
        "avg_reward": float(mean(reward_values)) if reward_values else None,
        "avg_steps": float(mean(step_values)) if step_values else None,
    }


def selected_retrieval_rows(record: Dict[str, Any], max_learnings: Optional[int]) -> List[Dict[str, Any]]:
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
        count = max_learnings if max_learnings is not None else len(base)
    return [row for row in base[:count] if isinstance(row, dict)]


def valid_level_ok(level: str, min_valid_level: Optional[str]) -> bool:
    if not min_valid_level:
        return True
    return VALID_LEVEL_PRIORITY.get(level or "CANDIDATE", 0) >= VALID_LEVEL_PRIORITY.get(min_valid_level, 0)


def count_help_steps(trajectory: Dict[str, Any]) -> int:
    count = 0
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip().lower()
        if step.get("is_help_call") is True or action.startswith("help[") or action.startswith("help ["):
            count += 1
    return count


def action_leakage_reason(action: Any) -> Optional[str]:
    text = str(action or "")
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith(("Action:", "Obs:", "Observation:")):
        return "action starts with prompt label"
    if LABEL_LEAKAGE_RE.search(stripped):
        return "action contains prompt label continuation"
    if NEXT_ACTION_RE.search(stripped):
        return "action contains a second action continuation"
    if "```" in stripped:
        return "action contains code fence"
    if "\n" in stripped and not SQL_START_RE.match(stripped):
        return "non-SQL action spans multiple lines"
    if SQL_START_RE.match(stripped) and re.search(r"(?im)\n\s*submit\b", stripped):
        return "SQL action includes submit continuation"
    return None


def extract_kb_task_ids(memory_bank_path: Optional[Path]) -> Tuple[set[str], List[JsonIssue]]:
    issues: List[JsonIssue] = []
    task_ids: set[str] = set()
    if not memory_bank_path:
        return task_ids, issues
    if not memory_bank_path.exists():
        issues.append(JsonIssue(str(memory_bank_path), "Memory bank path does not exist"))
        return task_ids, issues
    payload = load_json(memory_bank_path, issues)
    if not isinstance(payload, list):
        if payload is not None:
            issues.append(JsonIssue(str(memory_bank_path), "Expected memory bank to contain a list"))
        return task_ids, issues
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        for key in ("task_id", "task_id_str", "source_task_id"):
            value = entry.get(key)
            if value:
                task_ids.add(str(value))
        for ref_key in ("issue_ref", "evidence_ref"):
            ref = entry.get(ref_key)
            if isinstance(ref, dict):
                for key in ("task_id", "issue_task_id", "evidence_task_id"):
                    value = ref.get(key)
                    if value:
                        task_ids.add(str(value))
        for key in ("issue_task_id", "evidence_task_id"):
            value = entry.get(key)
            if value:
                task_ids.add(str(value))
    return task_ids, issues


def audit_runs(
    runs: List[RunData],
    memory_bank_path: Optional[Path],
    reward_threshold: float,
    min_valid_level: Optional[str],
    max_learnings: Optional[int],
    split_counts: Dict[str, int],
    default_count: Optional[int],
    strict_provenance: bool,
    run_command: str,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    def add_error(check: str, run: Optional[RunData], detail: str, **extra: Any) -> None:
        item = {"check": check, "detail": detail}
        if run is not None:
            item.update({
                "run_dir": str(run.run_dir),
                "split": run.split,
                "framework": run.framework,
                "seed": run.seed,
            })
        item.update(extra)
        errors.append(item)

    def add_warning(check: str, run: Optional[RunData], detail: str, **extra: Any) -> None:
        item = {"check": check, "detail": detail}
        if run is not None:
            item.update({
                "run_dir": str(run.run_dir),
                "split": run.split,
                "framework": run.framework,
                "seed": run.seed,
            })
        item.update(extra)
        warnings.append(item)

    for run in runs:
        for missing in run.missing_files:
            add_error("missing_json", run, f"Missing required artifact: {missing}")
        for issue in run.json_issues:
            add_error("corrupt_json", run, issue.detail, path=issue.path)

    kb_task_ids, kb_issues = extract_kb_task_ids(memory_bank_path)
    for issue in kb_issues:
        add_error("memory_bank_json", None, issue.detail, path=issue.path)

    run_metrics: List[Dict[str, Any]] = []
    duplicate_task_runs = 0
    reward_checked = 0
    reward_missing = 0
    help_checked = 0
    retrieved_checked = 0

    eval_task_ids: set[str] = set()
    for run in runs:
        rows = final_records(run)
        ids = [task_id(row) for row in rows]
        blank_ids = sum(1 for tid in ids if not tid)
        if blank_ids:
            add_error("task_ids", run, f"{blank_ids} final records are missing task_id")

        counts = Counter(tid for tid in ids if tid)
        duplicates = {tid: count for tid, count in counts.items() if count > 1}
        if duplicates:
            duplicate_task_runs += 1
            add_error("duplicate_task_ids", run, "Duplicate final task IDs", duplicates=duplicates)

        expected = expected_task_count(run, split_counts, default_count)
        if expected is not None and len(counts) != expected:
            add_error(
                "task_count",
                run,
                f"Expected {expected} unique tasks, found {len(counts)}",
                expected=expected,
                observed=len(counts),
            )

        if run.split == "train":
            add_error("held_out_split", run, "Evaluation run is on split=train")

        eval_task_ids.update(counts)
        metrics = recompute_metrics(run, reward_threshold)
        metrics.update({"split": run.split, "framework": run.framework, "seed": run.seed, "run_dir": str(run.run_dir)})
        run_metrics.append(metrics)

        for row in rows:
            tid = task_id(row)
            raw_reward_present = "reward" in row and row.get("reward") not in (None, "")
            reward = as_float(row.get("reward"))
            success = as_bool(row.get("success"))
            if reward is None:
                if raw_reward_present:
                    add_error("reward_range", run, f"Non-finite or non-numeric reward for task {tid}", task_id=tid, reward=row.get("reward"))
                reward_missing += 1
                continue
            reward_checked += 1
            if reward < -REWARD_EPS or reward > 1.0 + REWARD_EPS:
                add_warning("reward_range", run, f"Reward outside nominal [0, 1] range for task {tid}", task_id=tid, reward=reward)
            if success is not None and success != (reward >= reward_threshold):
                add_error(
                    "success_reward_consistency",
                    run,
                    f"Success flag does not equal reward >= {reward_threshold} for task {tid}",
                    task_id=tid,
                    success=success,
                    reward=reward,
                )

        for trajectory in run.trajectories:
            tid = task_id(trajectory)
            step_help = count_help_steps(trajectory)
            logged_count = trajectory.get("help_call_count")
            if logged_count is not None:
                help_checked += 1
                try:
                    logged_int = int(logged_count)
                except (TypeError, ValueError):
                    add_error("help_call_counts", run, f"Non-integer help_call_count for task {tid}", task_id=tid)
                    logged_int = step_help
                if logged_int != step_help:
                    add_error(
                        "help_call_counts",
                        run,
                        f"help_call_count does not match help steps for task {tid}",
                        task_id=tid,
                        logged_count=logged_int,
                        counted_from_steps=step_help,
                    )
            help_calls = trajectory.get("help_calls")
            if isinstance(help_calls, list) and len(help_calls) != step_help:
                add_error(
                    "help_call_counts",
                    run,
                    f"help_calls list length does not match help steps for task {tid}",
                    task_id=tid,
                    help_calls_len=len(help_calls),
                    counted_from_steps=step_help,
                )

            context = trajectory.get("context_from_retrieval")
            if isinstance(context, list):
                if max_learnings is not None and len(context) > max_learnings:
                    add_error(
                        "max_learnings",
                        run,
                        f"Retrieved context exceeds max_learnings for task {tid}",
                        task_id=tid,
                        retrieved=len(context),
                        max_learnings=max_learnings,
                    )
                for index, memory in enumerate(context):
                    if not isinstance(memory, dict):
                        continue
                    retrieved_checked += 1
                    level = str(memory.get("valid_level", "CANDIDATE") or "CANDIDATE")
                    if not valid_level_ok(level, min_valid_level):
                        add_error(
                            "retrieved_valid_level",
                            run,
                            f"Context memory below min_valid_level for task {tid}",
                            task_id=tid,
                            row_index=index,
                            valid_level=level,
                            min_valid_level=min_valid_level,
                        )

            for call_index, call in enumerate(trajectory.get("help_calls") or []):
                if not isinstance(call, dict):
                    continue
                response = str(call.get("response") or "")
                levels = HELP_VALIDATION_RE.findall(response)
                if step_help and not levels and "No relevant" not in response and "disabled" not in response.lower():
                    add_warning(
                        "retrieved_valid_level",
                        run,
                        f"Could not parse validation levels from help response for task {tid}",
                        task_id=tid,
                        help_call_index=call_index,
                    )
                for level in levels:
                    retrieved_checked += 1
                    if not valid_level_ok(level, min_valid_level):
                        add_error(
                            "retrieved_valid_level",
                            run,
                            f"Help memory below min_valid_level for task {tid}",
                            task_id=tid,
                            help_call_index=call_index,
                            valid_level=level,
                            min_valid_level=min_valid_level,
                        )

            action_format_noise_reported = False
            for step in trajectory.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                reason = action_leakage_reason(step.get("action"))
                if reason:
                    expected_sql_format_noise = (
                        reason == "action contains code fence"
                        and run.split == "test"
                        and run.framework == "react_cr"
                        and tid == "sql_288"
                        and (
                            as_bool(trajectory.get("success")) is True
                            or (as_float(trajectory.get("reward")) or 0.0) >= reward_threshold
                        )
                    )
                    if expected_sql_format_noise:
                        if not action_format_noise_reported:
                            add_warning(
                                "action_format_noise",
                                run,
                                "Recovered bare code-fence action for sql_288; environment rejected it and the task succeeded",
                                task_id=tid,
                                action=str(step.get("action") or "")[:500],
                            )
                            action_format_noise_reported = True
                        continue
                    add_error(
                        "action_continuation_leakage",
                        run,
                        f"{reason} for task {tid}",
                        task_id=tid,
                        step=step.get("step"),
                        action=str(step.get("action") or "")[:500],
                    )

        for retrieval_task_id, record in run.retrieval_records.items():
            selected_rows = selected_retrieval_rows(record, max_learnings)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            actual_selected = metadata.get("actual_selected_count")
            if isinstance(actual_selected, int) and max_learnings is not None and actual_selected > max_learnings:
                add_error(
                    "max_learnings",
                    run,
                    "Retrieval metadata actual_selected_count exceeds max_learnings",
                    task_id=retrieval_task_id,
                    actual_selected_count=actual_selected,
                    max_learnings=max_learnings,
                )
            for index, memory in enumerate(selected_rows):
                retrieved_checked += 1
                level = str(memory.get("valid_level", "CANDIDATE") or "CANDIDATE")
                if not valid_level_ok(level, min_valid_level):
                    add_error(
                        "retrieved_valid_level",
                        run,
                        "Selected retrieval row below min_valid_level",
                        task_id=retrieval_task_id,
                        row_index=index,
                        valid_level=level,
                        min_valid_level=min_valid_level,
                    )

    overlap = sorted(kb_task_ids & eval_task_ids)
    if overlap:
        add_error(
            "train_eval_overlap",
            None,
            f"{len(overlap)} evaluation task IDs also appear in the KB provenance",
            examples=overlap[:20],
            overlap_count=len(overlap),
        )

    by_split_seed: Dict[Tuple[str, str], Dict[str, set[str]]] = defaultdict(dict)
    for run in runs:
        by_split_seed[(run.split, run.seed)][run.framework] = set(final_record_map(run))
    for (split, seed), framework_tasks in sorted(by_split_seed.items()):
        if not framework_tasks:
            continue
        baseline_name = "react" if "react" in framework_tasks else sorted(framework_tasks)[0]
        baseline = framework_tasks[baseline_name]
        for framework, ids in sorted(framework_tasks.items()):
            if framework == baseline_name:
                continue
            missing = sorted(baseline - ids)
            extra = sorted(ids - baseline)
            if missing or extra:
                add_error(
                    "task_alignment",
                    None,
                    f"Task IDs for {framework} do not align with {baseline_name} on split={split}, seed={seed}",
                    split=split,
                    seed=seed,
                    framework=framework,
                    baseline=baseline_name,
                    missing_from_framework=missing[:20],
                    extra_in_framework=extra[:20],
                    missing_count=len(missing),
                    extra_count=len(extra),
                )

    config = runs[0].suite_config if runs else {}
    provenance_fields = {
        "model": config.get("model"),
        "embedding_provider": config.get("embedding_provider"),
        "splits": config.get("splits"),
        "frameworks": config.get("frameworks"),
        "max_learnings": max_learnings if max_learnings is not None else config.get("max_learnings"),
        "min_valid_level": min_valid_level if min_valid_level is not None else config.get("min_valid_level"),
        "memory_bank": str(memory_bank_path) if memory_bank_path else config.get("memory_bank"),
        "seed_values": sorted({run.seed for run in runs}),
        "run_command": run_command or config.get("command") or config.get("run_command"),
    }
    for key, value in provenance_fields.items():
        if value in (None, "", [], {}):
            target = add_error if strict_provenance else add_warning
            target("provenance", None, f"Missing provenance field: {key}", field=key)

    stats["runs"] = len(runs)
    stats["run_metrics"] = run_metrics
    stats["unique_eval_tasks"] = len(eval_task_ids)
    stats["kb_task_ids"] = len(kb_task_ids)
    stats["duplicate_task_runs"] = duplicate_task_runs
    stats["reward_records_checked"] = reward_checked
    stats["reward_records_missing"] = reward_missing
    stats["help_records_checked"] = help_checked
    stats["retrieved_memories_checked"] = retrieved_checked

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "provenance": provenance_fields,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a final LTM run directory")
    parser.add_argument("run_root", type=Path, help="Suite root containing suite_config.json and split/framework/seed dirs")
    parser.add_argument("--kb-path", type=Path, default=None, help="Memory KB path used by the run")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON audit report path")
    parser.add_argument("--expected-task-count", action="append", default=[], help="Expected count, or split=count. Repeatable")
    parser.add_argument("--reward-threshold", type=float, default=None, help="Override success reward threshold")
    parser.add_argument("--min-valid-level", default=None, choices=["", "CANDIDATE", "VALID_SAME_TRIAL", "VALID_NEXT_TRIAL"], help="Override minimum valid level")
    parser.add_argument("--max-learnings", type=int, default=None, help="Override context retrieval cap")
    parser.add_argument("--run-command", default="", help="Original run command for provenance if not stored in suite_config")
    parser.add_argument("--strict-provenance", action="store_true", help="Treat missing provenance fields as errors")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    config_path = run_root / "suite_config.json"
    config_issues: List[JsonIssue] = []
    suite_config = load_json(config_path, config_issues) if config_path.exists() else {}
    if not isinstance(suite_config, dict):
        suite_config = {}

    split_counts, default_count = parse_expected_counts(args.expected_task_count)
    reward_threshold = (
        args.reward_threshold
        if args.reward_threshold is not None
        else float(suite_config.get("reward_threshold", 1.0) or 1.0)
    )
    min_valid_level = args.min_valid_level if args.min_valid_level is not None else suite_config.get("min_valid_level")
    max_learnings = args.max_learnings if args.max_learnings is not None else suite_config.get("max_learnings")
    if max_learnings is not None:
        max_learnings = int(max_learnings)

    kb_path = args.kb_path
    if kb_path is None and suite_config.get("memory_bank"):
        kb_path = Path(str(suite_config["memory_bank"]))
    if kb_path is not None and not kb_path.is_absolute():
        kb_path = (Path.cwd() / kb_path).resolve()

    run_dirs = discover_run_dirs(run_root)
    runs = [load_run_data(run_root, run_dir, suite_config) for run_dir in run_dirs]
    if config_issues:
        placeholder = RunData(run_root, run_root, "", "", "", suite_config, json_issues=config_issues)
        runs.insert(0, placeholder)
    if not config_path.exists():
        placeholder = RunData(run_root, run_root, "", "", "", suite_config, missing_files=["suite_config.json"])
        runs.insert(0, placeholder)

    report = audit_runs(
        runs=runs,
        memory_bank_path=kb_path,
        reward_threshold=reward_threshold,
        min_valid_level=min_valid_level,
        max_learnings=max_learnings,
        split_counts=split_counts,
        default_count=default_count,
        strict_provenance=args.strict_provenance,
        run_command=args.run_command,
    )
    report["audit"] = {
        "script": str(Path(__file__).resolve()),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "git_commit": repo_commit(Path(__file__).resolve().parents[2]),
        "run_root": str(run_root),
        "suite_config": str(config_path),
    }

    if args.output:
        write_json(args.output, report)

    print(json.dumps({
        "passed": report["passed"],
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
        "runs": report["stats"]["runs"],
        "unique_eval_tasks": report["stats"]["unique_eval_tasks"],
    }, indent=2, sort_keys=True))
    if report["errors"]:
        for item in report["errors"][:20]:
            print(f"ERROR [{item['check']}]: {item['detail']}", file=sys.stderr)
        if len(report["errors"]) > 20:
            print(f"... {len(report['errors']) - 20} additional errors omitted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
