#!/usr/bin/env python3
"""
Run a standardized InterCode SQL framework suite on dev/test splits.

Frameworks:
1. ReAct
2. ReAct + Reflexion (max N trials)
3. ReAct + Context Retrieval (CR)
4. ReAct + Tool Retrieval (TR)
5. ReAct + CR + TR
6. ReAct + hard_neg CR + TR

Mirrors scripts/run_framework_suite.py but adapted for InterCode SQL.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence
import csv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FRAMEWORK_ORDER = [
    "react",
    "react_reflexion",
    "react_cr",
    "react_tr",
    "react_cr_tr",
    "react_hard_neg_cr_tr",
]

FRAMEWORK_DISPLAY = {
    "react": "ReAct",
    "react_reflexion": "React + Reflexion (final trials, max 7)",
    "react_cr": "ReAct + Context Retrieval (CR)",
    "react_tr": "ReAct + Tool Retrieval (TR)",
    "react_cr_tr": "react + CR + TR",
    "react_hard_neg_cr_tr": "react + hard_neg CR + TR",
}


@dataclass
class TaskInfo:
    split: str
    task_index: int       # Index into InterCode dataset
    task_id_str: str      # String ID for logging


def parse_list_arg(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_prompts(prompts_path: Path) -> Dict[str, str]:
    with open(prompts_path, "r") as f:
        return json.load(f)


def load_split_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r") as f:
        return json.load(f)


def discover_tasks_for_split(
    data_dir: Path, split: str, num_tasks: int
) -> List[TaskInfo]:
    """Load task indices from split manifest."""
    manifest_path = data_dir / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}\n"
            f"Run: python scripts/setup_intercode_sql.py first."
        )

    manifest = load_split_manifest(manifest_path)
    task_indices = manifest.get("task_indices", [])

    # Limit to requested count
    task_indices = task_indices[:num_tasks]

    return [
        TaskInfo(
            split=split,
            task_index=idx,
            task_id_str=f"sql_{idx}",
        )
        for idx in task_indices
    ]


def extract_task_desc(observation: str) -> str:
    """Extract the task question from an InterCode SQL observation."""
    from src.envs.intercode_sql_env import InterCodeSQLEnv
    return InterCodeSQLEnv.get_task_description(observation)


def count_action_steps(history_items: List[Dict[str, str]]) -> int:
    return sum(1 for item in history_items if item.get("label") == "action")


def make_env(task_index: int, data_path: str, image_name: str = "docker-env-sql-spider"):
    """Create an InterCodeSQLEnv for a specific task."""
    from src.envs.intercode_sql_env import InterCodeSQLEnv
    return InterCodeSQLEnv(
        task_index=task_index,
        data_path=data_path,
        image_name=image_name,
    )


def build_sql_instructions() -> str:
    return (
        "You are an AI agent interacting with a MySQL database to answer questions using SQL.\n"
        "You can execute SQL queries as actions. The observation will show the query results.\n"
        "Use SHOW TABLES and SHOW COLUMNS FROM <table> to explore the schema.\n"
        "To finish, first execute the SQL query whose result answers the question, then issue "
        "the action exactly as 'submit' on its own line.\n"
        "Do not put the answer, SQL, punctuation, or explanation after 'submit'; the runner "
        "grades the most recent SQL result.\n"
        "Use think: to reason about the problem before acting.\n\n"
        "Here are some examples:\n\n"
    )


def ensure_clean_dir(path: Path, resume: bool = False) -> None:
    if resume:
        path.mkdir(parents=True, exist_ok=True)
        return
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_completed_task_ids(run_dir: Path) -> set:
    """Read world.log to find task IDs already completed (for resume)."""
    completed = set()
    world_log = run_dir / "world.log"
    if not world_log.exists():
        return completed
    import re
    # Match lines like "Task #123: sql_598 (idx=598) - SUCCESS" or "- FAIL"
    pattern = re.compile(r"Task #\d+: (sql_\d+)")
    with open(world_log, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                completed.add(m.group(1))
    return completed


def load_completed_reflexion_tasks(run_dir: Path) -> set:
    """For reflexion resume: find tasks that succeeded or were attempted in all trials."""
    completed = set()
    world_log = run_dir / "world.log"
    if not world_log.exists():
        return completed
    import re
    # Find tasks that had SUCCESS in any trial
    pattern = re.compile(r"Task #\d+ Trial #\d+: (SUCCESS|FAIL)")
    task_pattern = re.compile(r"Task #(\d+) Trial")
    # We need task_id mapping, easier to just track task indices done in last trial
    # For simplicity: return set of task indices (as ints) that appear in world.log
    indices = set()
    idx_pattern = re.compile(r"Task #(\d+) Trial")
    with open(world_log, "r") as f:
        for line in f:
            m = idx_pattern.search(line)
            if m:
                indices.add(int(m.group(1)))
    return indices


def compute_metrics(
    task_ids: Sequence[str],
    attempt_records: Sequence[Dict[str, Any]],
    final_success_override: Optional[int] = None,
    reward_threshold: float = 1.0,
) -> Dict[str, Any]:
    total_tasks = len(task_ids)
    if final_success_override is not None:
        success_count = final_success_override
    else:
        # InterCode SQL rewards are continuous/partial-match. A logged agent
        # boolean may mean "non-zero reward" for generic agents, so SQL metrics
        # must use the canonical exact-match threshold.
        success_count = sum(
            1 for r in attempt_records
            if float(r.get("reward", 0) or 0) >= reward_threshold
        )

    accuracy = (success_count / total_tasks) if total_tasks else 0.0

    # Exclude reconstructed-from-log entries from step statistics (step_num=0
    # in those entries would bias averages downward — see resume bug fix).
    step_records = [r for r in attempt_records if not r.get("reconstructed", False)]
    avg_steps_per_trial = (
        mean(float(r.get("step_num", 0)) for r in step_records)
        if step_records else 0.0
    )
    avg_reward = (
        mean(float(r.get("reward", 0)) for r in attempt_records)
        if attempt_records else 0.0
    )

    steps_by_task = {task_id: 0.0 for task_id in task_ids}
    counted_tasks: set = set()
    for row in step_records:
        task_id = row.get("task_id")
        if task_id not in steps_by_task:
            continue
        steps_by_task[task_id] += float(row.get("step_num", 0))
        counted_tasks.add(task_id)

    # Only average over tasks that actually have step data
    if counted_tasks:
        avg_steps_per_task = mean(steps_by_task[t] for t in counted_tasks)
    else:
        avg_steps_per_task = 0.0

    return {
        "success": int(success_count),
        "total": int(total_tasks),
        "success_total": f"{int(success_count)} / {int(total_tasks)}",
        "accuracy": float(accuracy),
        "avg_reward": float(avg_reward),
        "avg_steps_per_trial": float(avg_steps_per_trial),
        "avg_steps_per_task": float(avg_steps_per_task),
        "attempt_count": int(len(attempt_records)),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_react_baseline(
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    data_path: str,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
) -> Dict[str, Any]:
    from src.frameworks.react import ReAct

    agent = ReAct(model=model, to_print=not quiet)
    attempts: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    world_log = run_dir / "world.log"

    # Resume: load existing trajectories and skip completed tasks
    completed_ids = set()
    if resume:
        completed_ids = load_completed_task_ids(run_dir)
        traj_path = run_dir / "trajectories.json"
        if traj_path.exists():
            with open(traj_path, "r") as f:
                trajectories = json.load(f)
            for t in trajectories:
                attempts.append({
                    "task_id": t["task_id"], "task_index": t["task_index"],
                    "trial_num": 1, "step_num": t.get("step_num", 0),
                    "success": t["success"], "reward": t.get("reward", 0),
                })
        elif completed_ids:
            # No trajectories.json but world.log exists — reconstruct attempts from world.log.
            # step_num is unknown for these reconstructed entries; mark them as
            # reconstructed=True so compute_metrics can exclude them from step averages.
            import re
            wlog = run_dir / "world.log"
            if wlog.exists():
                pat = re.compile(r"Task #\d+: (sql_\d+) \(idx=(\d+)\) - (SUCCESS|FAIL) \(reward=([\d.]+)\)")
                with open(wlog, "r") as f:
                    for line in f:
                        m = pat.search(line)
                        if m:
                            tid, tidx, result, rew = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
                            attempts.append({
                                "task_id": tid, "task_index": tidx,
                                "trial_num": 1, "step_num": 0,
                                "success": result == "SUCCESS", "reward": rew,
                                "reconstructed": True,
                            })
                            # Also add stub trajectory entry (no steps, but has metadata)
                            trajectories.append({
                                "task_id": tid, "task_index": tidx,
                                "task_desc": "", "trial_num": 1,
                                "steps": [], "success": result == "SUCCESS",
                                "step_num": 0, "reward": rew,
                                "reconstructed": True,
                            })
        if completed_ids:
            print(f"  Resuming: skipping {len(completed_ids)} already-completed tasks")

    # Build base prompt from all SQL few-shot examples
    all_prompts = []
    for key in sorted(prompts.keys()):
        if key.startswith("intercode_sql_react_"):
            all_prompts.append(prompts[key])
    base_prompt = "\n\n".join(all_prompts) if all_prompts else ""

    # Prepend SQL-specific instructions
    base_prompt = build_sql_instructions() + base_prompt

    log_mode = "a" if resume else "w"
    with open(world_log, log_mode) as wf:
        wf.write("ReAct baseline run (InterCode SQL)\n")

    for i, task in enumerate(task_infos):
        if task.task_id_str in completed_ids:
            continue

        success = False
        reward = 0.0
        step_num = 0
        history_items: List[Dict[str, str]] = []

        task_desc = ""
        env = make_env(task.task_index, data_path)
        try:
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob)
            history, _agent_success = agent.run(
                env=env,
                base_prompt=base_prompt,
                memory=[],
                start_ob=ob,
            )
            # InterCode: done=True just means "submit" was called.
            # Use the actual reward to determine success.
            reward = env.last_reward
            success = reward >= reward_threshold
            history_items = history.to_json()
            step_num = count_action_steps(history_items)
        except Exception as e:
            # Always log to stderr so silent-exception paths are visible even
            # under --quiet. Diagnostic for the trajectory-gap issue.
            print(f"[react] Task failed with error ({task.task_id_str}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            success = False
            reward = 0.0
            step_num = 0
            history_items = []
        finally:
            env.close()

        attempts.append(
            {
                "task_id": task.task_id_str,
                "task_index": task.task_index,
                "trial_num": 1,
                "step_num": step_num,
                "success": success,
                "reward": reward,
            }
        )
        trajectories.append(
            {
                "task_id": task.task_id_str,
                "task_index": task.task_index,
                "task_type": "intercode_sql",
                "task_desc": task_desc,
                "trial_num": 1,
                "steps": history_items,
                "success": success,
                "step_num": step_num,
                "reward": reward,
                "split": task.split,
            }
        )

        with open(world_log, "a") as wf:
            wf.write(
                f"Task #{i}: {task.task_id_str} (idx={task.task_index}) - "
                f"{'SUCCESS' if success else 'FAIL'} (reward={reward:.2f})\n"
            )

    with open(run_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    metrics = compute_metrics(
        [t.task_id_str for t in task_infos], attempts, reward_threshold=reward_threshold
    )
    return metrics


def load_completed_memory_task_ids(run_dir: Path) -> set:
    """Read agent_trajectories.jsonl to find task_ids already completed (for memory variant resume)."""
    completed = set()
    traj_jsonl = run_dir / "agent_trajectories.jsonl"
    if not traj_jsonl.exists():
        return completed
    with open(traj_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                tid = entry.get("task_id")
                if tid:
                    completed.add(tid)
            except json.JSONDecodeError:
                pass
    return completed


def run_memory_agent_variant(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    data_path: str,
    memory_bank_path: Path,
    reward_threshold: float,
    quiet: bool,
    max_learnings: int = 25,
    min_valid_level: str = "",
    resume: bool = False,
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_agent import MemoryAgent
    from src.frameworks.memory_retrieval_v2.agents.hard_neg_memory_agent import HardNegMemoryAgent

    class ContextOnlyAgent(MemoryAgent):
        def _get_help_instructions(self) -> str:
            return ""

        def _execute_help_tool(self, query: str) -> str:
            return "Help tool is disabled for this Context-Only run."

    if framework_id == "react_cr":
        agent = ContextOnlyAgent(model=model, to_print=not quiet, env_kind="intercode_sql")
        include_context = True
    elif framework_id == "react_tr":
        agent = MemoryAgent(model=model, to_print=not quiet, env_kind="intercode_sql")
        include_context = False
    elif framework_id == "react_cr_tr":
        agent = MemoryAgent(model=model, to_print=not quiet, env_kind="intercode_sql")
        include_context = True
    elif framework_id == "react_hard_neg_cr_tr":
        agent = HardNegMemoryAgent(model=model, to_print=not quiet, env_kind="intercode_sql")
        include_context = True
    else:
        raise ValueError(f"Unsupported memory variant: {framework_id}")

    # Build base prompt
    all_prompts = []
    for key in sorted(prompts.keys()):
        if key.startswith("intercode_sql_react_"):
            all_prompts.append(prompts[key])
    base_prompt_examples = "\n\n".join(all_prompts) if all_prompts else ""

    base_prompt = build_sql_instructions() + base_prompt_examples

    world_log = run_dir / "world.log"

    # Resume: detect already-completed task_ids from agent_trajectories.jsonl.
    # Build attempts from the existing file so summary metrics include old results.
    completed_ids: set = set()
    attempts: List[Dict[str, Any]] = []
    if resume:
        completed_ids = load_completed_memory_task_ids(run_dir)
        if completed_ids:
            print(f"  Resuming: skipping {len(completed_ids)} already-completed tasks")
            # Reconstruct attempt stubs from existing trajectories so metrics
            # computed at the end reflect the full (old + new) result set.
            # Use seen_ids to deduplicate: take only the first occurrence per
            # task_id so duplicate lines in the JSONL don't bias metrics.
            traj_jsonl = run_dir / "agent_trajectories.jsonl"
            seen_ids: set = set()
            with open(traj_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tid = entry.get("task_id")
                        if tid and tid not in seen_ids:
                            seen_ids.add(tid)
                            attempts.append({
                                "task_id": tid,
                                "task_index": entry.get("task_index", 0),
                                "trial_num": 1,
                                "step_num": entry.get("step_num", 0),
                                "success": bool(entry.get("success", False)),
                                "reward": float(entry.get("reward", 0)),
                            })
                    except json.JSONDecodeError:
                        pass

    log_mode = "a" if resume else "w"
    with open(world_log, log_mode) as wf:
        wf.write(f"{framework_id} run (InterCode SQL)\n")
        wf.write(f"memory_bank={memory_bank_path}\n")

    for i, task in enumerate(task_infos):
        if task.task_id_str in completed_ids:
            continue

        success = False
        reward = 0.0
        step_num = 0
        env = make_env(task.task_index, data_path)
        try:
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob)

            history, _agent_success = agent.run(
                env=env,
                base_prompt=base_prompt,
                memory=[],
                start_ob=ob,
                task_id=task.task_id_str,
                trial_num=1,
                log_dir=str(run_dir),
                task_desc=(task_desc if include_context else ""),
                memory_bank_path=str(memory_bank_path),
                max_learnings=max_learnings,
                min_valid_level=min_valid_level,
                split=task.split,
            )
            reward = env.last_reward
            success = reward >= reward_threshold
            step_num = count_action_steps(history.to_json())
        except Exception as e:
            # Always log to stderr — see note in run_react_baseline.
            print(f"[{framework_id}] Task failed with error ({task.task_id_str}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            success = False
            reward = 0.0
            step_num = 0
        finally:
            env.close()

        attempts.append(
            {
                "task_id": task.task_id_str,
                "task_index": task.task_index,
                "trial_num": 1,
                "step_num": step_num,
                "success": success,
                "reward": reward,
            }
        )
        with open(world_log, "a") as wf:
            wf.write(
                f"Task #{i}: {task.task_id_str} - "
                f"{'SUCCESS' if success else 'FAIL'} (reward={reward:.2f})\n"
            )

    metrics = compute_metrics(
        [t.task_id_str for t in task_infos], attempts, reward_threshold=reward_threshold
    )
    with open(world_log, "a") as wf:
        wf.write("-----\n")
        wf.write(f"SUCCESS: {metrics['success']}\n")
        wf.write(f"FAIL: {metrics['total'] - metrics['success']}\n")
        wf.write(f"TOTAL: {metrics['total']}\n")
        wf.write(f"ACCURACY: {metrics['accuracy']:.4f}\n")
        wf.write("-----\n")
    return metrics


def run_reflexion(
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    max_trials: int,
    prompts: Dict[str, str],
    data_path: str,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_allocation import MemoryAllocationReflexion

    agent = MemoryAllocationReflexion(model=model, to_print=not quiet)
    world_log = run_dir / "world.log"

    # Build base prompt
    all_prompts = []
    for key in sorted(prompts.keys()):
        if key.startswith("intercode_sql_react_"):
            all_prompts.append(prompts[key])
    base_prompt_examples = "\n\n".join(all_prompts) if all_prompts else ""

    base_prompt = build_sql_instructions() + base_prompt_examples

    state: Dict[str, Dict[str, Any]] = {
        task.task_id_str: {"memory": [], "is_success": False, "skip": False}
        for task in task_infos
    }

    # Resume: reconstruct state from existing trajectories.
    # We track (task_id, trial_num) pairs that already have a trajectory entry
    # so we can skip them in any trial, not just trial 0.
    resumed_task_trial_pairs: set = set()  # (task_id_str, trial_num_1based)
    if resume:
        traj_jsonl_path = run_dir / "trajectories.jsonl"
        if traj_jsonl_path.exists():
            with open(traj_jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tid = entry.get("task_id", "")
                        tnum = int(entry.get("trial_num", 0))
                        if tid and tnum:
                            resumed_task_trial_pairs.add((tid, tnum))
                        if tid in state and entry.get("success"):
                            state[tid]["is_success"] = True
                    except json.JSONDecodeError:
                        pass
        # Also load reflexions for memory
        reflexions_path = run_dir / "reflexions.json"
        if reflexions_path.exists():
            with open(reflexions_path, "r") as f:
                reflexions = json.load(f)
            for entry in reflexions:
                tid = entry.get("task_id", "")
                if tid in state:
                    reflexion = entry.get("reflexion", "")
                    if reflexion:
                        state[tid]["memory"].append(reflexion)
        resumed_success = sum(1 for s in state.values() if s["is_success"])
        print(
            f"  Resuming reflexion: {len(resumed_task_trial_pairs)} (task,trial) pairs already done, "
            f"{resumed_success} tasks already succeeded"
        )

    log_mode = "a" if resume else "w"
    with open(world_log, log_mode) as wf:
        wf.write("Reflexion run (InterCode SQL)\n")
        wf.write(f"max_trials={max_trials}\n")

    trials_executed = 0

    for trial_idx in range(max_trials):
        trials_executed += 1
        trial_successes = 0
        trial_failures = 0

        with open(world_log, "a") as wf:
            wf.write(f"\n\n***** Start Trial #{trial_idx} *****\n\n")

        for i, task in enumerate(task_infos):
            task_state = state[task.task_id_str]
            if task_state["is_success"] or task_state["skip"]:
                trial_successes += 1
                continue

            # Resume: skip tasks already attempted in this (task, trial) pair
            if resume and (task.task_id_str, trial_idx + 1) in resumed_task_trial_pairs:
                continue

            success = False
            reward = 0.0
            history_items = []
            task_desc = ""
            env = make_env(task.task_index, data_path)
            try:
                ob, _info = env.reset()
                task_desc = extract_task_desc(ob)

                history, _agent_success = agent.run(
                    env=env,
                    base_prompt=base_prompt,
                    memory=task_state["memory"],
                    start_ob=ob,
                    task_id=task.task_id_str,
                    trial_num=trial_idx + 1,
                    log_dir=str(run_dir),
                    task_desc=task_desc,
                )
                reward = env.last_reward
                success = reward >= reward_threshold
                history_items = history.to_json()
            except Exception as e:
                if not quiet:
                    print(f"[reflexion] Task failed ({task.task_id_str}): {e}")
                task_state["skip"] = True
                success = False
            finally:
                env.close()

            # Log trajectory for KB construction
            step_num = count_action_steps(history_items)
            traj_entry = {
                "task_id": task.task_id_str,
                "task_index": task.task_index,
                "task_type": "intercode_sql",
                "task_desc": task_desc,
                "trial_num": trial_idx + 1,
                "steps": history_items,
                "success": success,
                "reward": reward,
                "step_num": step_num,
                "split": task.split,
            }
            traj_jsonl_path = run_dir / "trajectories.jsonl"
            with open(traj_jsonl_path, "a") as f:
                f.write(json.dumps(traj_entry, separators=(",", ":")) + "\n")

            task_state["is_success"] = task_state["is_success"] or success
            if success:
                trial_successes += 1
            else:
                trial_failures += 1

            with open(world_log, "a") as wf:
                wf.write(
                    f"Task #{i} Trial #{trial_idx}: {'SUCCESS' if success else 'FAIL'} "
                    f"(reward={reward:.2f})\n"
                )

        # Load reflexions from the agent's output
        reflexions_path = run_dir / "reflexions.json"
        if reflexions_path.exists():
            with open(reflexions_path, "r") as f:
                reflexions = json.load(f)
            for task in task_infos:
                task_state = state[task.task_id_str]
                if task_state["is_success"] or task_state["skip"]:
                    continue
                for entry in reversed(reflexions):
                    if (
                        entry.get("task_id") == task.task_id_str
                        and int(entry.get("trial_num", -1)) == trial_idx + 1
                    ):
                        reflexion = entry.get("reflexion", "")
                        if reflexion:
                            task_state["memory"].append(reflexion)
                        break

        accuracy = trial_successes / len(task_infos) if task_infos else 0.0
        with open(world_log, "a") as wf:
            wf.write("-----\n")
            wf.write(f"SUCCESS: {trial_successes}\n")
            wf.write(f"FAIL: {trial_failures}\n")
            wf.write(f"TOTAL: {len(task_infos)}\n")
            wf.write(f"ACCURACY: {accuracy:.2f}\n")
            wf.write(f"\n***** End Trial #{trial_idx} *****\n\n")

        if all(item["is_success"] or item["skip"] for item in state.values()):
            break

    # Collect attempt records from trajectory JSONL logs
    traj_jsonl_path = run_dir / "trajectories.jsonl"
    attempt_records: List[Dict[str, Any]] = []
    all_trajectories: List[Dict[str, Any]] = []
    if traj_jsonl_path.exists():
        with open(traj_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    all_trajectories.append(entry)
                    attempt_records.append(
                        {
                            "task_id": entry.get("task_id"),
                            "trial_num": int(entry.get("trial_num", 0)),
                            "step_num": int(entry.get("step_num", 0)),
                            "success": bool(entry.get("success", False)),
                            "reward": float(entry.get("reward", 0)),
                        }
                    )
                except json.JSONDecodeError:
                    pass

    # Also write a consolidated trajectories.json for KB construction
    with open(run_dir / "trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2)

    final_success = sum(1 for item in state.values() if item["is_success"])
    metrics = compute_metrics(
        [t.task_id_str for t in task_infos],
        attempt_records,
        final_success_override=final_success,
        reward_threshold=reward_threshold,
    )
    metrics["trials_executed"] = trials_executed
    return metrics


def write_split_summaries(
    split: str,
    rows: List[Dict[str, Any]],
    summaries_dir: Path,
) -> None:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    json_path = summaries_dir / f"summary_{split}.json"
    csv_path = summaries_dir / f"summary_{split}.csv"
    md_path = summaries_dir / f"summary_{split}.md"

    base_accuracy = 0.0
    for row in rows:
        if row["framework_id"] == "react":
            base_accuracy = float(row["accuracy"])
            break

    for row in rows:
        row["gain_from_base_react"] = float(row["accuracy"]) - base_accuracy

    with open(json_path, "w") as f:
        json.dump({"split": split, "rows": rows}, f, indent=2)

    csv_columns = [
        "framework",
        "success_total",
        "accuracy",
        "avg_reward",
        "avg_steps_per_trial",
        "avg_steps_per_task",
        "gain_from_base_react",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": row["framework"],
                    "success_total": row["success_total"],
                    "accuracy": f"{row['accuracy']:.4f}",
                    "avg_reward": f"{row.get('avg_reward', 0):.4f}",
                    "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                    "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                    "gain_from_base_react": f"{row['gain_from_base_react']:.4f}",
                }
            )

    with open(md_path, "w") as f:
        f.write(f"# Summary ({split})\n\n")
        f.write(
            "| Framework | Success / Total | Accuracy | Avg Reward | "
            "Avg Steps Per Trial | Avg Steps per Task | Gain from Base ReAct |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['framework']} | {row['success_total']} | {row['accuracy']:.4f} | "
                f"{row.get('avg_reward', 0):.4f} | "
                f"{row['avg_steps_per_trial']:.4f} | {row['avg_steps_per_task']:.4f} | "
                f"{row['gain_from_base_react']:+.4f} |\n"
            )


def write_combined_summary(all_rows: List[Dict[str, Any]], summaries_dir: Path) -> None:
    csv_path = summaries_dir / "summary_all.csv"
    json_path = summaries_dir / "summary_all.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "framework", "success_total", "accuracy", "avg_reward",
                "avg_steps_per_trial", "avg_steps_per_task", "gain_from_base_react",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "framework": row["framework"],
                    "success_total": row["success_total"],
                    "accuracy": f"{row['accuracy']:.4f}",
                    "avg_reward": f"{row.get('avg_reward', 0):.4f}",
                    "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                    "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                    "gain_from_base_react": f"{row['gain_from_base_react']:.4f}",
                }
            )

    with open(json_path, "w") as f:
        json.dump({"rows": all_rows}, f, indent=2)


def run_framework(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    max_trials: int,
    prompts: Dict[str, str],
    data_path: str,
    memory_bank_path: Path,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
    max_learnings: int = 25,
    min_valid_level: str = "",
) -> Dict[str, Any]:
    if framework_id == "react":
        return run_react_baseline(
            task_infos=task_infos,
            run_dir=run_dir,
            model=model,
            prompts=prompts,
            data_path=data_path,
            reward_threshold=reward_threshold,
            quiet=quiet,
            resume=resume,
        )
    if framework_id == "react_reflexion":
        return run_reflexion(
            task_infos=task_infos,
            run_dir=run_dir,
            model=model,
            max_trials=max_trials,
            prompts=prompts,
            data_path=data_path,
            reward_threshold=reward_threshold,
            quiet=quiet,
            resume=resume,
        )
    if framework_id in {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}:
        return run_memory_agent_variant(
            framework_id=framework_id,
            task_infos=task_infos,
            run_dir=run_dir,
            model=model,
            prompts=prompts,
            data_path=data_path,
            memory_bank_path=memory_bank_path,
            reward_threshold=reward_threshold,
            quiet=quiet,
            resume=resume,
            max_learnings=max_learnings,
            min_valid_level=min_valid_level,
        )
    raise ValueError(f"Unknown framework id: {framework_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InterCode SQL framework suite")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=200,
        help="Number of tasks per split (default: 200)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="dev,test",
        help="Comma-separated splits (default: dev,test)",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default=",".join(FRAMEWORK_ORDER),
        help="Comma-separated framework IDs to run",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=7,
        help="Max trials for Reflexion run (default: 7)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Chat model name (default: gemini-2.5-flash). Use 'claude-haiku-4-5' for Claude.",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="",
        choices=["", "gemini", "openai"],
        help=(
            "Embedding provider for retrieval (default: auto-pair). "
            "Auto-pair: gemini-* chat models → gemini, claude-* chat models → openai. "
            "Override explicitly with 'gemini' or 'openai'."
        ),
    )
    parser.add_argument(
        "--memory-bank",
        type=str,
        default=(
            "intercode_sql_runs/memory_agent_runs/train/react_reflexion/"
            "knowledge_base.sql_sanitized.json"
        ),
        help=(
            "Path to memory bank JSON. Default points to the sanitized SQL KB "
            "created by scripts/utils/sql_memory_sanitizer.py."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/intercode_sql",
        help="Directory with split manifests",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="intercode_sql_runs/memory_agent_runs",
        help="Root output directory",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=1.0,
        help="Reward threshold for success (default: 1.0 = exact match)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-step framework print output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previously interrupted run. Skips already-completed tasks "
            "detected from existing trajectory files (agent_trajectories.jsonl for "
            "memory variants; trajectories.json for react baseline); appends to "
            "world.log rather than wiping it. Supported for: react, react_cr, "
            "react_tr, react_cr_tr, react_hard_neg_cr_tr. Reflexion resume is "
            "also supported via trajectories.jsonl."
        ),
    )
    parser.add_argument(
        "--max-learnings",
        type=int,
        default=25,
        help="Max learnings to retrieve for context (default: 25).",
    )
    parser.add_argument(
        "--min-valid-level",
        type=str,
        default="",
        choices=["", "CANDIDATE", "VALID_SAME_TRIAL", "VALID_NEXT_TRIAL"],
        help="Minimum validation level for retrieved learnings (default: no filter).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for task ordering and any harness-level sampling (default: 0). "
            "Note: LLM calls use temperature=0 but are not bit-deterministic across "
            "providers; the seed primarily varies harness sampling/ordering."
        ),
    )
    args = parser.parse_args()

    # Seed the harness RNG for deterministic task ordering and any sampling.
    # LLM calls at temperature=0 are not bit-deterministic across providers,
    # so the seed primarily affects harness-level randomness (task shuffling,
    # random sampling if any). Use different seeds (0, 1, 2) for multi-seed runs.
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    # Set embedding provider before any retrieval module is imported.
    # Default: ALWAYS Gemini (the embedding model used for KB extraction).
    # Mixing chat models is fine for cross-lab; mixing embedding models would
    # silently change retrieval and confound the comparison.
    def _auto_embedding_provider(model: str) -> str:
        return "gemini"

    embedding_provider = args.embedding_provider if args.embedding_provider else _auto_embedding_provider(args.model)
    os.environ["LTM_EMBEDDING_PROVIDER"] = embedding_provider
    print(f"Using model={args.model}, embedding_provider={embedding_provider}, seed={args.seed}")

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = (base_dir / args.data_dir).resolve()
    runs_root = (base_dir / args.runs_root).resolve()
    memory_bank_path = (base_dir / args.memory_bank).resolve()
    prompts_path = base_dir / "data" / "intercode_sql" / "prompts" / "intercode_sql_prompts.json"

    selected_splits = parse_list_arg(args.splits)
    selected_frameworks = parse_list_arg(args.frameworks)

    for framework_id in selected_frameworks:
        if framework_id not in FRAMEWORK_DISPLAY:
            raise ValueError(f"Unsupported framework: {framework_id}")

    # Check memory bank for memory-based frameworks
    memory_frameworks = {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}
    if memory_frameworks & set(selected_frameworks) and not memory_bank_path.exists():
        print(f"Warning: Memory bank not found at {memory_bank_path}")
        print(
            "Memory-based frameworks will fail. Run KB construction first, then:\n"
            "  python3 scripts/utils/sql_memory_sanitizer.py"
        )

    runs_root.mkdir(parents=True, exist_ok=True)
    summaries_dir = runs_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path)

    # Get data_path from first split manifest
    first_manifest = load_split_manifest(data_dir / f"{selected_splits[0]}.json")
    data_path = first_manifest.get("data_path", "")
    if not data_path or not Path(data_path).exists():
        print(f"Error: InterCode SQL data not found at: {data_path}")
        print("Run scripts/setup_intercode_sql.py first.")
        sys.exit(1)

    # Save suite config
    save_json(
        runs_root / "suite_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "num_tasks": args.num_tasks,
            "splits": selected_splits,
            "frameworks": selected_frameworks,
            "max_trials": args.max_trials,
            "model": args.model,
            "embedding_provider": embedding_provider,
            "memory_bank": str(memory_bank_path),
            "max_learnings": args.max_learnings,
            "min_valid_level": args.min_valid_level,
            "data_path": data_path,
            "reward_threshold": args.reward_threshold,
            "seed": args.seed,
        },
    )

    # Each seed run gets its own sub-directory so parallel seed runs don't
    # overwrite each other: runs_root/<split>/<framework_id>/seed_<N>/
    seed_tag = f"seed_{args.seed}"

    all_rows: List[Dict[str, Any]] = []

    for split in selected_splits:
        task_infos = discover_tasks_for_split(data_dir, split, args.num_tasks)
        if not task_infos:
            print(f"No valid tasks for split={split}; skipping.")
            continue

        split_rows: List[Dict[str, Any]] = []

        for framework_id in selected_frameworks:
            run_dir = runs_root / split / framework_id / seed_tag
            ensure_clean_dir(run_dir, resume=args.resume)

            print(
                f"Running split={split} framework={framework_id} seed={args.seed} "
                f"tasks={len(task_infos)} -> {run_dir}"
            )
            metrics = run_framework(
                framework_id=framework_id,
                task_infos=task_infos,
                run_dir=run_dir,
                model=args.model,
                max_trials=args.max_trials,
                prompts=prompts,
                data_path=data_path,
                memory_bank_path=memory_bank_path,
                reward_threshold=args.reward_threshold,
                quiet=args.quiet,
                resume=args.resume,
                max_learnings=args.max_learnings,
                min_valid_level=args.min_valid_level,
            )

            row = {
                "split": split,
                "framework_id": framework_id,
                "framework": FRAMEWORK_DISPLAY[framework_id],
                "success_total": metrics["success_total"],
                "accuracy": metrics["accuracy"],
                "avg_reward": metrics.get("avg_reward", 0),
                "avg_steps_per_trial": metrics["avg_steps_per_trial"],
                "avg_steps_per_task": metrics["avg_steps_per_task"],
                "attempt_count": metrics["attempt_count"],
                "run_dir": str(run_dir),
            }
            split_rows.append(row)

            save_json(
                run_dir / "metrics.json",
                {
                    "framework_id": framework_id,
                    "framework": row["framework"],
                    "split": split,
                    "num_tasks": len(task_infos),
                    "metrics": metrics,
                },
            )

        write_split_summaries(split, split_rows, summaries_dir)
        all_rows.extend(split_rows)

    # Add gain columns to combined summary
    rows_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_split[row["split"]].append(row)

    for split, rows in rows_by_split.items():
        base_accuracy = 0.0
        for row in rows:
            if row["framework_id"] == "react":
                base_accuracy = float(row["accuracy"])
                break
        for row in rows:
            row["gain_from_base_react"] = float(row["accuracy"]) - base_accuracy

    write_combined_summary(all_rows, summaries_dir)
    print(f"\nSuite run complete. Summaries written to: {summaries_dir}")


if __name__ == "__main__":
    main()
