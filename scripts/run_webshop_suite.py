#!/usr/bin/env python3
"""
Run a standardized WebShop framework suite on webshop_mini splits.

Frameworks:
1. ReAct
2. ReAct + Reflexion (max N trials)
3. ReAct + Context Retrieval (CR)
4. ReAct + Tool Retrieval (TR)
5. ReAct + CR + TR
6. ReAct + hard_neg CR + TR

Mirrors scripts/run_framework_suite.py but adapted for WebShop.

Changes from scripts/legacy/run_webshop_suite.py:
  - Added --seed (default 0); random.seed + numpy.random.seed; seed_tag in run_dir paths.
  - Added --embedding-provider arg + _auto_embedding_provider(); sets
    LTM_EMBEDDING_PROVIDER env var before any retrieval import.
  - MemoryAgent / HardNegMemoryAgent now constructed with env_kind="webshop".
  - agent.run() calls pass split=task.split for trajectory recording.
  - traj dicts in run_react_baseline, run_memory_agent_variant, and run_reflexion
    include "split": task.split.
  - suite_config.json records seed and embedding_provider.
  - Output directories are seed-aware: runs_root/<split>/<framework_id>/seed_<N>/
  - base_dir resolves one level up (scripts/ is one deep from project root).
  - Removed unused --misc-archive-root default that pointed at legacy webshop_runs.
"""

import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


def atomic_json_dump(data, filepath: Path, indent=2):
    """Write JSON atomically: write to temp file, then rename."""
    dirpath = filepath.parent
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, filepath)
    except:
        os.unlink(tmp_path)
        raise


def append_jsonl(entry: dict, filepath: Path):
    """Append a single JSON object as one line to a JSONL file (append-only backup)."""
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with open(filepath, "a") as f:
        f.write(line)


def recover_trajectories_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Recover trajectories from JSONL backup file."""
    trajectories = []
    if not jsonl_path.exists():
        return trajectories
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trajectories.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[recovery] Skipping corrupted line {line_num} in {jsonl_path}")
    return trajectories


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
    task_id: int          # WebShop uses integer task IDs
    task_id_str: str      # String version for logging


def parse_list_arg(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_prompts(prompts_path: Path) -> Dict[str, str]:
    with open(prompts_path, "r") as f:
        return json.load(f)


def archive_existing_runs(runs_root: Path, misc_archive_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    entries = list(runs_root.iterdir())
    if not entries:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = misc_archive_root / f"archive_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        shutil.move(str(entry), str(archive_dir / entry.name))
    return archive_dir


def discover_tasks_for_split(
    manifest_dir: Path, split: str, num_tasks: int, start_idx: int = 0
) -> List[TaskInfo]:
    """Load task IDs from webshop_mini manifest."""
    manifest_path = manifest_dir / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"WebShop mini manifest not found: {manifest_path}\n"
            f"Run: python scripts/create_webshop_mini.py first."
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    task_ids = manifest.get("task_ids", [])

    # Limit to requested count with offset
    task_ids = task_ids[start_idx : start_idx + num_tasks]

    return [
        TaskInfo(
            split=split,
            task_id=tid,
            task_id_str=f"webshop_{tid}",
        )
        for tid in task_ids
    ]


def extract_task_desc(observation: str) -> str:
    """Extract the task instruction from a WebShop observation."""
    from src.envs.webshop_env import WebShopEnv
    return WebShopEnv.get_task_instruction(observation)


def count_action_steps(history_items: List[Dict[str, str]]) -> int:
    return sum(1 for item in history_items if item.get("label") == "action")


def make_env(task_id: int, webshop_path: str = None, num_products: int = None, server=None):
    """Create a WebShopEnv for a specific task, optionally sharing a SimServer."""
    from src.envs.webshop_env import WebShopEnv
    return WebShopEnv(
        task_id=task_id,
        webshop_path=webshop_path,
        num_products=num_products,
        server=server,
    )


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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
        success_count = sum(
            1 for r in attempt_records
            if r.get("success") or r.get("reward", 0) >= reward_threshold
        )

    accuracy = (success_count / total_tasks) if total_tasks else 0.0
    avg_steps_per_trial = (
        mean(float(r.get("step_num", 0)) for r in attempt_records)
        if attempt_records else 0.0
    )
    avg_reward = (
        mean(float(r.get("reward", 0)) for r in attempt_records)
        if attempt_records else 0.0
    )

    steps_by_task = {task_id: 0.0 for task_id in task_ids}
    for row in attempt_records:
        task_id = row.get("task_id")
        if task_id not in steps_by_task:
            continue
        steps_by_task[task_id] += float(row.get("step_num", 0))

    avg_steps_per_task = mean(steps_by_task.values()) if steps_by_task else 0.0

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
    atomic_json_dump(payload, path)


def run_react_baseline(
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    webshop_path: str,
    num_products: int,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
    max_workers: int = 1,
    shared_server=None,
) -> Dict[str, Any]:
    from src.frameworks.react import ReAct

    attempts: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    completed_task_ids: set = set()
    world_log = run_dir / "world.log"
    lock = threading.Lock()

    if resume:
        attempts_path = run_dir / "attempts.json"
        traj_path = run_dir / "trajectories.json"
        if attempts_path.exists():
            try:
                with open(attempts_path, "r") as f:
                    attempts = json.load(f)
                completed_task_ids = {a["task_id"] for a in attempts}
            except json.JSONDecodeError:
                recovered = recover_trajectories_from_jsonl(run_dir / "trajectories.jsonl")
                if recovered:
                    attempts = [{"task_id": e["task_id"], "reward": e.get("reward", 0), "success": e.get("success", False)} for e in recovered]
                    completed_task_ids = {a["task_id"] for a in attempts}
                    atomic_json_dump(attempts, attempts_path)
                    print(f"[react] Recovered {len(attempts)} attempts from JSONL backup")
        if traj_path.exists():
            try:
                with open(traj_path, "r") as f:
                    trajectories = json.load(f)
            except json.JSONDecodeError:
                trajectories = recover_trajectories_from_jsonl(run_dir / "trajectories.jsonl")
                if trajectories:
                    atomic_json_dump(trajectories, traj_path)
                    print(f"[react] Recovered {len(trajectories)} trajectories from JSONL backup")
        print(f"[resume] {len(completed_task_ids)} tasks already completed for react baseline")
        with open(world_log, "a") as wf:
            wf.write(f"\n--- Resumed at {datetime.now().isoformat()} ---\n")
    else:
        with open(world_log, "w") as wf:
            wf.write("ReAct baseline run (WebShop)\n")

    # Combine all few-shot examples
    all_prompts = []
    for key in sorted(prompts.keys()):
        if key.startswith("webshop_react_"):
            all_prompts.append(prompts[key])
    base_prompt = "\n\n".join(all_prompts) if all_prompts else prompts.get("webshop_react_0", "")

    # Filter to pending tasks
    pending = [(i, t) for i, t in enumerate(task_infos) if t.task_id_str not in completed_task_ids]
    if not pending:
        return compute_metrics([t.task_id_str for t in task_infos], attempts, reward_threshold=reward_threshold)

    # Use pre-loaded shared server, or load once as fallback
    primary_env = None
    if shared_server is None:
        print(f"[react] Loading WebShop environment...")
        primary_env = make_env(task_infos[0].task_id, webshop_path, num_products)
        shared_server = primary_env.get_server()
    print(f"[react] Running {len(pending)} tasks with {max_workers} workers")

    def run_single_task(task_idx: int, task: TaskInfo):
        # Each thread gets its own agent and env (lightweight with shared server)
        agent = ReAct(model=model, to_print=not quiet)
        env = make_env(task.task_id, webshop_path, num_products=None, server=shared_server)

        success = False
        reward = 0.0
        step_num = 0
        task_desc = ""
        history_items = []

        try:
            env.task_id = task.task_id
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob)
            history, raw_success = agent.run(
                env=env, base_prompt=base_prompt, memory=[], start_ob=ob,
            )
            history_items = history.to_json()
            step_num = count_action_steps(history_items)

            reward = 0.0
            if history_items:
                last_obs = history_items[-1].get("value", "")
                if "Your score (min 0.0, max 1.0) [SEP]" in last_obs:
                    try:
                        score_str = last_obs.split("Your score (min 0.0, max 1.0) [SEP]")[1].split("[SEP]")[0].strip()
                        reward = float(score_str)
                    except ValueError:
                        pass
            success = raw_success or (reward >= reward_threshold)
        except Exception as e:
            if not quiet:
                print(f"[react] Task failed with error ({task.task_id_str}): {e}")

        attempt = {
            "task_id": task.task_id_str, "task_id_int": task.task_id,
            "trial_num": 1, "step_num": step_num, "reward": reward, "success": success,
        }
        traj = {
            "task_id": task.task_id_str, "task_id_int": task.task_id,
            "task_desc": task_desc, "trial_num": 1, "steps": history_items,
            "success": success, "reward": reward, "step_num": step_num,
            "split": task.split,
        }

        with lock:
            attempts.append(attempt)
            trajectories.append(traj)
            atomic_json_dump(attempts, run_dir / "attempts.json")
            atomic_json_dump(trajectories, run_dir / "trajectories.json")
            append_jsonl(traj, run_dir / "trajectories.jsonl")
            with open(world_log, "a") as wf:
                status = "SUCCESS" if success else "FAIL"
                wf.write(f"Task #{task_idx}: {task.task_id_str} - {status} (reward={reward:.4f}, steps={step_num})\n")

        return attempt

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_task, i, t): t for i, t in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                task = futures[future]
                print(f"[react] Worker crashed for {task.task_id_str}: {e}")

    if primary_env:
        primary_env.close()

    metrics = compute_metrics(
        [t.task_id_str for t in task_infos], attempts, reward_threshold=reward_threshold
    )
    return metrics


def run_memory_agent_variant(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    memory_bank_path: Path,
    webshop_path: str,
    num_products: int,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
    max_workers: int = 1,
    shared_server=None,
    max_learnings: int = 25,
    min_valid_level: str = "",
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_agent import MemoryAgent
    from src.frameworks.memory_retrieval_v2.agents.hard_neg_memory_agent import HardNegMemoryAgent

    # Fix #5: env_kind="webshop" for webshop-flavored help instructions.
    class ContextOnlyAgent(MemoryAgent):
        def _get_help_instructions(self) -> str:
            return ""
        def _execute_help_tool(self, query: str) -> str:
            return "Help tool is disabled for this Context-Only run."

    if framework_id == "react_cr":
        include_context = True
    elif framework_id == "react_tr":
        include_context = False
    elif framework_id == "react_cr_tr":
        include_context = True
    elif framework_id == "react_hard_neg_cr_tr":
        include_context = True
    else:
        raise ValueError(f"Unsupported memory variant: {framework_id}")

    def _make_agent():
        if framework_id == "react_cr":
            return ContextOnlyAgent(model=model, to_print=not quiet, env_kind="webshop")
        elif framework_id == "react_hard_neg_cr_tr":
            return HardNegMemoryAgent(model=model, to_print=not quiet, env_kind="webshop")
        else:
            return MemoryAgent(model=model, to_print=not quiet, env_kind="webshop")

    # Build base prompt
    all_prompts = []
    for key in sorted(prompts.keys()):
        if key.startswith("webshop_react_"):
            all_prompts.append(prompts[key])
    base_prompt = "\n\n".join(all_prompts) if all_prompts else ""

    attempts: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    completed_task_ids: set = set()
    world_log = run_dir / "world.log"
    lock = threading.Lock()

    if resume:
        attempts_path = run_dir / "attempts.json"
        traj_path = run_dir / "trajectories.json"
        if attempts_path.exists():
            try:
                with open(attempts_path, "r") as f:
                    attempts = json.load(f)
                completed_task_ids = {a["task_id"] for a in attempts}
            except json.JSONDecodeError:
                recovered = recover_trajectories_from_jsonl(run_dir / "trajectories.jsonl")
                if recovered:
                    attempts = [{"task_id": e["task_id"], "reward": e.get("reward", 0), "success": e.get("success", False)} for e in recovered]
                    completed_task_ids = {a["task_id"] for a in attempts}
                    atomic_json_dump(attempts, attempts_path)
                    print(f"[{framework_id}] Recovered {len(attempts)} attempts from JSONL backup")
        if traj_path.exists():
            try:
                with open(traj_path, "r") as f:
                    trajectories = json.load(f)
            except json.JSONDecodeError:
                trajectories = recover_trajectories_from_jsonl(run_dir / "trajectories.jsonl")
                if trajectories:
                    atomic_json_dump(trajectories, traj_path)
                    print(f"[{framework_id}] Recovered {len(trajectories)} trajectories from JSONL backup")
        print(f"[resume] {len(completed_task_ids)} tasks already completed for {framework_id}")
        with open(world_log, "a") as wf:
            wf.write(f"\n--- Resumed at {datetime.now().isoformat()} ---\n")
    else:
        with open(world_log, "w") as wf:
            wf.write(f"{framework_id} run (WebShop)\n")
            wf.write(f"memory_bank={memory_bank_path}\n")

    # Filter to pending tasks
    pending = [(i, t) for i, t in enumerate(task_infos) if t.task_id_str not in completed_task_ids]
    if not pending:
        return compute_metrics([t.task_id_str for t in task_infos], attempts, reward_threshold=reward_threshold)

    # Use pre-loaded shared server, or load once as fallback
    primary_env = None
    if shared_server is None:
        print(f"[{framework_id}] Loading WebShop environment...")
        primary_env = make_env(task_infos[0].task_id, webshop_path, num_products)
        shared_server = primary_env.get_server()
    print(f"[{framework_id}] Running {len(pending)} tasks with {max_workers} workers")

    # Pre-warm embedding caches before spawning threads to avoid race conditions
    if max_workers > 1:
        from src.frameworks.memory_retrieval_v2.retrieval.core.learning_counts import build_learning_counts_table
        from src.frameworks.memory_retrieval_v2.retrieval.core.embedding_cache import create_knowledge_base_embeddings
        mb_path = str(memory_bank_path)
        print(f"[{framework_id}] Pre-warming embedding caches for {mb_path}...")
        build_learning_counts_table(mb_path)
        create_knowledge_base_embeddings(mb_path, embed_field="issue_text")
        print(f"[{framework_id}] Embedding caches ready.")

    def run_single_task(task_idx: int, task: TaskInfo):
        agent = _make_agent()
        env = make_env(task.task_id, webshop_path, num_products=None, server=shared_server)

        success = False
        reward = 0.0
        step_num = 0
        task_desc = ""
        history_items = []

        try:
            env.task_id = task.task_id
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob) if include_context else ""

            # Fix #6: pass split=task.split so the agent's trajectory writer
            # captures it in the agent_trajectories.jsonl output.
            history, raw_success = agent.run(
                env=env, base_prompt=base_prompt, memory=[], start_ob=ob,
                task_id=task.task_id_str, trial_num=1, log_dir=str(run_dir),
                task_desc=task_desc, memory_bank_path=str(memory_bank_path),
                max_learnings=max_learnings, min_valid_level=min_valid_level,
                split=task.split,
            )
            history_items = history.to_json()
            step_num = count_action_steps(history_items)

            reward = 0.0
            if history_items:
                last_obs = history_items[-1].get("value", "")
                if "Your score (min 0.0, max 1.0) [SEP]" in last_obs:
                    try:
                        score_str = last_obs.split("Your score (min 0.0, max 1.0) [SEP]")[1].split("[SEP]")[0].strip()
                        reward = float(score_str)
                    except ValueError:
                        pass
            success = raw_success or (reward >= reward_threshold)
        except Exception as e:
            if not quiet:
                print(f"[{framework_id}] Task failed ({task.task_id_str}): {e}")

        attempt = {
            "task_id": task.task_id_str, "task_id_int": task.task_id,
            "trial_num": 1, "step_num": step_num, "reward": reward, "success": success,
        }
        traj = {
            "task_id": task.task_id_str, "task_id_int": task.task_id,
            "task_desc": task_desc, "trial_num": 1, "steps": history_items,
            "success": success, "reward": reward, "step_num": step_num,
            "split": task.split,
        }

        with lock:
            attempts.append(attempt)
            trajectories.append(traj)
            atomic_json_dump(attempts, run_dir / "attempts.json")
            atomic_json_dump(trajectories, run_dir / "trajectories.json")
            append_jsonl(traj, run_dir / "trajectories.jsonl")
            with open(world_log, "a") as wf:
                status = "SUCCESS" if success else "FAIL"
                wf.write(f"Task #{task_idx}: {task.task_id_str} - {status} (reward={reward:.4f}, steps={step_num})\n")

        return attempt

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_task, i, t): t for i, t in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                task = futures[future]
                print(f"[{framework_id}] Worker crashed for {task.task_id_str}: {e}")

    if primary_env:
        primary_env.close()

    metrics = compute_metrics(
        [t.task_id_str for t in task_infos], attempts, reward_threshold=reward_threshold
    )
    with open(world_log, "a") as wf:
        wf.write("-----\n")
        wf.write(f"SUCCESS: {metrics['success']}\n")
        wf.write(f"FAIL: {metrics['total'] - metrics['success']}\n")
        wf.write(f"TOTAL: {metrics['total']}\n")
        wf.write(f"ACCURACY: {metrics['accuracy']:.4f}\n")
        wf.write(f"AVG REWARD: {metrics['avg_reward']:.4f}\n")
        wf.write("-----\n")
    return metrics


def run_reflexion(
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    max_trials: int,
    prompts: Dict[str, str],
    webshop_path: str,
    num_products: int,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
    max_workers: int = 1,
    shared_server=None,
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_allocation import MemoryAllocationReflexion

    world_log = run_dir / "world.log"
    lock = threading.Lock()

    # Build base prompt
    all_prompts = []
    for key in sorted(prompts.keys()):
        if key.startswith("webshop_react_"):
            all_prompts.append(prompts[key])
    base_prompt = "\n\n".join(all_prompts) if all_prompts else ""

    state: Dict[str, Dict[str, Any]] = {
        task.task_id_str: {"memory": [], "is_success": False, "skip": False, "best_reward": 0.0}
        for task in task_infos
    }

    trajectories_path = run_dir / "trajectories.json"
    trajectories = []
    resume_trial = 0

    if resume and trajectories_path.exists():
        try:
            with open(trajectories_path, "r") as f:
                trajectories = json.load(f)
        except json.JSONDecodeError:
            print(f"[reflexion] trajectories.json corrupted, recovering from JSONL backup...")
            trajectories = recover_trajectories_from_jsonl(run_dir / "trajectories.jsonl")
            if trajectories:
                atomic_json_dump(trajectories, trajectories_path)
                print(f"[reflexion] Recovered {len(trajectories)} trajectories from backup")

        if trajectories:
            # Rebuild state from existing trajectories
            for entry in trajectories:
                tid = entry.get("task_id")
                if tid not in state:
                    continue
                reward = float(entry.get("reward", 0))
                success = bool(entry.get("success", False))
                state[tid]["best_reward"] = max(state[tid]["best_reward"], reward)
                if success or reward >= reward_threshold:
                    state[tid]["is_success"] = True

            # Reload reflexions into memory
            reflexions_path = run_dir / "reflexions.json"
            if reflexions_path.exists():
                try:
                    with open(reflexions_path, "r") as f:
                        reflexions = json.load(f)
                    for entry in reflexions:
                        tid = entry.get("task_id")
                        if tid in state and not state[tid]["is_success"]:
                            reflexion = entry.get("reflexion", "")
                            if reflexion:
                                state[tid]["memory"].append(reflexion)
                except json.JSONDecodeError:
                    pass

            # Determine where to resume
            last_trial = max(int(e.get("trial_num", 1)) for e in trajectories)
            last_trial_tasks = {
                e.get("task_id") for e in trajectories
                if int(e.get("trial_num", 1)) == last_trial
            }
            all_task_ids = {t.task_id_str for t in task_infos}
            needs_attempt = {tid for tid in all_task_ids if not state.get(tid, {}).get("is_success")}
            if needs_attempt.issubset(last_trial_tasks | {tid for tid, s in state.items() if s["is_success"] or s["skip"]}):
                resume_trial = last_trial
            else:
                resume_trial = last_trial - 1

            completed = sum(1 for s in state.values() if s["is_success"])
            print(f"[resume] Loaded {len(trajectories)} existing trajectory entries")
            print(f"[resume] {completed}/{len(task_infos)} tasks already succeeded")
            print(f"[resume] Resuming from trial {resume_trial}")

        with open(world_log, "a") as wf:
            wf.write(f"\n--- Resumed at {datetime.now().isoformat()} ---\n")
    else:
        with open(world_log, "w") as wf:
            wf.write("Reflexion run (WebShop)\n")
            wf.write(f"max_trials={max_trials}\n")

        if trajectories_path.exists():
            try:
                with open(trajectories_path, "r") as f:
                    trajectories = json.load(f)
            except json.JSONDecodeError:
                pass

    trials_executed = 0

    # Use pre-loaded shared server, or load once as fallback
    primary_env = None
    if shared_server is None and task_infos:
        print(f"[reflexion] Loading WebShop environment...")
        primary_env = make_env(task_infos[0].task_id, webshop_path, num_products)
        shared_server = primary_env.get_server()

    for trial_idx in range(resume_trial, max_trials):
        trials_executed += 1
        trial_successes = 0
        trial_failures = 0
        additional_successes = 0

        with open(world_log, "a") as wf:
            wf.write(f"\n\n***** Start Trial #{trial_idx} *****\n\n")

        # Collect tasks that need running this trial
        pending_tasks = []
        for i, task in enumerate(task_infos):
            task_state = state[task.task_id_str]
            if task_state["is_success"]:
                trial_successes += 1
            elif task_state["skip"]:
                trial_failures += 1
            else:
                already_done = any(
                    e.get("task_id") == task.task_id_str and int(e.get("trial_num", 0)) == trial_idx + 1
                    for e in trajectories
                )
                if already_done:
                    trial_failures += 1
                else:
                    pending_tasks.append((i, task))

        print(f"[reflexion] Trial {trial_idx}: {len(pending_tasks)} tasks to run, "
              f"{trial_successes} already succeeded, {max_workers} workers")

        def run_reflexion_task(task_idx: int, task: TaskInfo, trial_idx: int):
            task_state = state[task.task_id_str]
            agent = MemoryAllocationReflexion(model=model, to_print=not quiet)
            env = make_env(task.task_id, webshop_path, num_products=None, server=shared_server)

            success = False
            reward = 0.0
            history_items = []

            try:
                env.task_id = task.task_id
                ob, _info = env.reset()
                task_desc = extract_task_desc(ob)

                _history, raw_success = agent.run(
                    env=env, base_prompt=base_prompt,
                    memory=list(task_state["memory"]),  # copy to avoid race
                    start_ob=ob, task_id=task.task_id_str,
                    trial_num=trial_idx + 1, log_dir=str(run_dir),
                    task_desc=task_desc,
                )

                history_items = _history.to_json()
                reward = 0.0
                if history_items:
                    last_obs = history_items[-1].get("value", "")
                    if "Your score (min 0.0, max 1.0) [SEP]" in last_obs:
                        try:
                            score_str = last_obs.split("Your score (min 0.0, max 1.0) [SEP]")[1].split("[SEP]")[0].strip()
                            reward = float(score_str)
                        except ValueError:
                            pass
                success = raw_success or (reward >= reward_threshold)

                # Fix #4: include split in reflexion trajectory entries.
                traj_entry = {
                    "task_id": task.task_id_str, "task_type": task.task_id_str,
                    "task_desc": task_desc, "trial_num": trial_idx + 1,
                    "steps": [], "success": success, "reward": reward,
                    "step_num": len(history_items) // 2 + 1 if history_items else 0,
                    "split": task.split,
                }
                step_idx = 1
                for j in range(0, len(history_items), 2):
                    if j + 1 < len(history_items):
                        traj_entry["steps"].append({
                            "step": step_idx,
                            "action": history_items[j]["value"],
                            "observation": history_items[j+1]["value"],
                        })
                        step_idx += 1

                with lock:
                    trajectories.append(traj_entry)
                    save_json(run_dir / "trajectories.json", trajectories)
                    append_jsonl(traj_entry, run_dir / "trajectories.jsonl")

            except Exception as e:
                if not quiet:
                    print(f"[react_reflexion] Task failed ({task.task_id_str}): {e}")
                task_state["skip"] = True
                success = False

            task_state["is_success"] = task_state["is_success"] or success
            task_state["best_reward"] = max(task_state["best_reward"], reward)

            with lock:
                with open(world_log, "a") as wf:
                    status = "SUCCESS" if success else "FAIL"
                    wf.write(f"Task #{task_idx} Trial #{trial_idx}: {status} (reward={reward:.4f})\n")

            return success

        # Run pending tasks in parallel
        results = []
        if pending_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_reflexion_task, i, t, trial_idx): t
                    for i, t in pending_tasks
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        task = futures[future]
                        print(f"[react_reflexion] Worker crashed for {task.task_id_str}: {e}")
                        state[task.task_id_str]["skip"] = True
                        results.append(False)

        # Count results
        for r in results:
            if r:
                trial_successes += 1
                additional_successes += 1
            else:
                trial_failures += 1

        # Load reflexions if available (written by agent during run)
        reflexions_path = run_dir / "reflexions.json"
        if reflexions_path.exists():
            try:
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
            except json.JSONDecodeError:
                pass

        accuracy = trial_successes / len(task_infos) if task_infos else 0.0
        with open(world_log, "a") as wf:
            wf.write("-----\n")
            wf.write(f"SUCCESS: {trial_successes}\n")
            wf.write(f"ADDITIONAL SUCCESS: {additional_successes}\n")
            wf.write(f"FAIL: {trial_failures}\n")
            wf.write(f"TOTAL: {len(task_infos)}\n")
            wf.write(f"ACCURACY: {accuracy:.2f}\n")
            wf.write("-----\n")
            wf.write(f"\n\n***** End Trial #{trial_idx} *****\n\n")

        if all(item["is_success"] or item["skip"] for item in state.values()):
            break

    if primary_env:
        primary_env.close()

    # Collect attempt records from trajectories
    attempt_records: List[Dict[str, Any]] = []
    if trajectories_path.exists():
        try:
            with open(trajectories_path, "r") as f:
                trajectories = json.load(f)
        except json.JSONDecodeError:
            print(f"[reflexion] Warning: trajectories.json corrupted at final read, using in-memory data")
        for entry in trajectories:
            attempt_records.append({
                "task_id": entry.get("task_id"),
                "trial_num": int(entry.get("trial_num", 0)),
                "step_num": int(entry.get("step_num", 0)),
                "reward": float(entry.get("reward", 0)),
                "success": bool(entry.get("success", False)),
            })

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
            writer.writerow({
                "framework": row["framework"],
                "success_total": row["success_total"],
                "accuracy": f"{row['accuracy']:.4f}",
                "avg_reward": f"{row['avg_reward']:.4f}",
                "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                "gain_from_base_react": f"{row['gain_from_base_react']:.4f}",
            })

    with open(md_path, "w") as f:
        f.write(f"# WebShop Summary ({split})\n\n")
        f.write("| Framework | Success / Total | Accuracy | Avg Reward | Avg Steps/Trial | Avg Steps/Task | Gain from ReAct |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['framework']} | {row['success_total']} | {row['accuracy']:.4f} | "
                f"{row['avg_reward']:.4f} | {row['avg_steps_per_trial']:.4f} | "
                f"{row['avg_steps_per_task']:.4f} | {row['gain_from_base_react']:+.4f} |\n"
            )


def write_combined_summary(all_rows: List[Dict[str, Any]], summaries_dir: Path) -> None:
    csv_path = summaries_dir / "summary_all.csv"
    json_path = summaries_dir / "summary_all.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "framework", "success_total", "accuracy",
                "avg_reward", "avg_steps_per_trial", "avg_steps_per_task",
                "gain_from_base_react",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "split": row["split"],
                "framework": row["framework"],
                "success_total": row["success_total"],
                "accuracy": f"{row['accuracy']:.4f}",
                "avg_reward": f"{row['avg_reward']:.4f}",
                "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                "gain_from_base_react": f"{row['gain_from_base_react']:.4f}",
            })

    with open(json_path, "w") as f:
        json.dump({"rows": all_rows}, f, indent=2)


def run_framework(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    max_trials: int,
    prompts: Dict[str, str],
    memory_bank_path: Path,
    webshop_path: str,
    num_products: int,
    reward_threshold: float,
    quiet: bool,
    resume: bool = False,
    max_workers: int = 1,
    shared_server=None,
    max_learnings: int = 25,
    min_valid_level: str = "",
) -> Dict[str, Any]:
    if framework_id == "react":
        return run_react_baseline(
            task_infos=task_infos, run_dir=run_dir, model=model,
            prompts=prompts, webshop_path=webshop_path,
            num_products=num_products, reward_threshold=reward_threshold,
            quiet=quiet, resume=resume, max_workers=max_workers,
            shared_server=shared_server,
        )
    if framework_id == "react_reflexion":
        return run_reflexion(
            task_infos=task_infos, run_dir=run_dir, model=model,
            max_trials=max_trials, prompts=prompts,
            webshop_path=webshop_path, num_products=num_products,
            reward_threshold=reward_threshold, quiet=quiet,
            resume=resume, max_workers=max_workers,
            shared_server=shared_server,
        )
    if framework_id in {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}:
        return run_memory_agent_variant(
            framework_id=framework_id, task_infos=task_infos,
            run_dir=run_dir, model=model, prompts=prompts,
            memory_bank_path=memory_bank_path, webshop_path=webshop_path,
            num_products=num_products, reward_threshold=reward_threshold,
            quiet=quiet, resume=resume, max_workers=max_workers,
            shared_server=shared_server,
            max_learnings=max_learnings, min_valid_level=min_valid_level,
        )
    raise ValueError(f"Unknown framework id: {framework_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WebShop framework suite")
    parser.add_argument(
        "--num-tasks", type=int, default=200,
        help="Number of tasks per split (default: 200)",
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="Start index for task list to evaluate (default: 0)",
    )
    parser.add_argument(
        "--splits", type=str, default="dev,test",
        help="Comma-separated splits (default: dev,test)",
    )
    parser.add_argument(
        "--frameworks", type=str, default=",".join(FRAMEWORK_ORDER),
        help="Comma-separated framework IDs to run",
    )
    parser.add_argument(
        "--max-trials", type=int, default=7,
        help="Max trials for Reflexion run (default: 7)",
    )
    # Fix #1: --model with correct default
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Chat model name (default: gemini-2.5-flash). Use 'claude-haiku-4-5' for Claude.",
    )
    # Fix #2: --embedding-provider
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="",
        choices=["", "gemini", "openai"],
        help=(
            "Embedding provider for retrieval (default: auto-pair). "
            "Auto-pair always returns 'gemini' so the retrieval distribution "
            "is consistent with the KB. Override only for fresh KB rebuilds."
        ),
    )
    parser.add_argument(
        "--memory-bank", type=str,
        default="webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train/knowledge_base.json",
        help="Path to memory bank JSON",
    )
    parser.add_argument(
        "--runs-root", type=str,
        default="webshop_runs/memory_retrieval_v2/memory_agent_runs",
        help="Root output directory",
    )
    parser.add_argument(
        "--misc-archive-root", type=str,
        default="webshop_runs/memory_retrieval_v2/misc/old_runs/memory_agent_runs",
        help="Archive directory root for old runs",
    )
    parser.add_argument(
        "--manifest-dir", type=str, default="webshop_mini",
        help="Directory containing webshop_mini manifests (default: webshop_mini)",
    )
    parser.add_argument(
        "--webshop-path", type=str,
        default=os.getenv("WEBSHOP_PATH", "./webshop"),
        help="Path to cloned WebShop repo",
    )
    parser.add_argument(
        "--num-products", type=int, default=None,
        help="Number of products to load (default: all)",
    )
    parser.add_argument(
        "--reward-threshold", type=float, default=1.0,
        help="Reward threshold for success (default: 1.0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing partial results instead of starting fresh",
    )
    parser.add_argument(
        "--archive-existing", action="store_true",
        help="Archive current runs-root contents before running",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Only prepare manifests/directories, do not execute",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of concurrent worker threads (default: 16)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce per-step framework print output",
    )
    parser.add_argument(
        "--max-learnings", type=int, default=7,
        help="Max learnings to retrieve for context (default: 7 for WebShop)",
    )
    parser.add_argument(
        "--min-valid-level", type=str, default="VALID_SAME_TRIAL",
        choices=["", "CANDIDATE", "VALID_SAME_TRIAL", "VALID_NEXT_TRIAL"],
        help="Minimum validation level for retrieved learnings (default: VALID_SAME_TRIAL)",
    )
    # Fix #3: --seed
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

    # Fix #3: seed the harness RNG.
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    # Fix #2: set embedding provider before any retrieval module is imported.
    # Default: ALWAYS use Gemini embeddings regardless of chat model — the
    # knowledge bases were extracted with Gemini embeddings and retrieval must
    # use the same model to stay distribution-consistent. Mixing chat models for
    # the cross-lab study is fine; mixing embedding models confounds retrieval.
    def _auto_embedding_provider(model: str) -> str:
        return "gemini"

    embedding_provider = args.embedding_provider if args.embedding_provider else _auto_embedding_provider(args.model)
    os.environ["LTM_EMBEDDING_PROVIDER"] = embedding_provider
    print(f"Using model={args.model}, embedding_provider={embedding_provider}, seed={args.seed}")

    # Fix: base_dir resolves to project root (scripts/ is one level down).
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / "webshop" / "prompts" / "webshop_prompts.json"
    runs_root = (base_dir / args.runs_root).resolve()
    misc_archive_root = (base_dir / args.misc_archive_root).resolve()
    memory_bank_path = (base_dir / args.memory_bank).resolve()
    manifest_dir = (base_dir / args.manifest_dir).resolve()

    selected_splits = parse_list_arg(args.splits)
    selected_frameworks = parse_list_arg(args.frameworks)

    for framework_id in selected_frameworks:
        if framework_id not in FRAMEWORK_DISPLAY:
            raise ValueError(f"Unsupported framework: {framework_id}")

    if args.archive_existing:
        archive_dir = archive_existing_runs(runs_root, misc_archive_root)
        if archive_dir:
            print(f"Archived existing run contents to: {archive_dir}")
        else:
            print("No existing run contents to archive.")

    runs_root.mkdir(parents=True, exist_ok=True)
    summaries_dir = runs_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    # Discover tasks per split
    all_split_tasks: Dict[str, List[TaskInfo]] = {}
    for split in selected_splits:
        task_infos = discover_tasks_for_split(manifest_dir, split, args.num_tasks, args.start_idx)
        all_split_tasks[split] = task_infos
        print(f"Split '{split}': {len(task_infos)} tasks (starting at index {args.start_idx})")

    # Fix #3: persist suite config with seed + embedding_provider.
    seed_tag = f"seed_{args.seed}"
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
            "seed": args.seed,
            "prepare_only": args.prepare_only,
        },
    )

    if args.prepare_only:
        print("Prepare-only mode. Exiting.")
        return

    # Load WebShop environment once for the entire suite
    print(f"Loading WebShop environment (single load for all frameworks)...")
    first_task = None
    for split in selected_splits:
        if all_split_tasks[split]:
            first_task = all_split_tasks[split][0]
            break
    if first_task is None:
        print("No tasks found. Exiting.")
        return

    primary_env = make_env(first_task.task_id, args.webshop_path, args.num_products)
    shared_server = primary_env.get_server()
    print(f"WebShop environment loaded. Server shared across all frameworks.\n")

    # Run frameworks
    all_rows: List[Dict[str, Any]] = []

    for split in selected_splits:
        task_infos = all_split_tasks[split]
        split_rows: List[Dict[str, Any]] = []

        for framework_id in selected_frameworks:
            framework_display = FRAMEWORK_DISPLAY[framework_id]
            # Fix #3: seed-aware output directory.
            run_dir = runs_root / split / framework_id / seed_tag
            if args.resume:
                run_dir.mkdir(parents=True, exist_ok=True)
            elif args.start_idx == 0:
                ensure_clean_dir(run_dir)
            else:
                run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Running: {framework_display} on {split} ({len(task_infos)} tasks)")
            print(f"Output:  {run_dir}")
            print(f"{'='*60}\n")

            try:
                metrics = run_framework(
                    framework_id=framework_id,
                    task_infos=task_infos,
                    run_dir=run_dir,
                    model=args.model,
                    max_trials=args.max_trials,
                    prompts=prompts,
                    memory_bank_path=memory_bank_path,
                    webshop_path=args.webshop_path,
                    num_products=args.num_products,
                    reward_threshold=args.reward_threshold,
                    quiet=args.quiet,
                    resume=args.resume,
                    max_workers=args.workers,
                    shared_server=shared_server,
                    max_learnings=args.max_learnings,
                    min_valid_level=args.min_valid_level,
                )
            except Exception as e:
                import traceback
                print(f"\n!!! Framework {framework_display} on {split} FAILED: {e}")
                traceback.print_exc()
                print(f"!!! Continuing with remaining frameworks...\n")
                continue

            save_json(run_dir / "metrics.json", metrics)

            row = {
                "split": split,
                "framework_id": framework_id,
                "framework": framework_display,
                **metrics,
            }
            split_rows.append(row)
            all_rows.append(row)

            print(f"\n  Results: {metrics['success_total']} success, "
                  f"accuracy={metrics['accuracy']:.4f}, "
                  f"avg_reward={metrics['avg_reward']:.4f}")

        write_split_summaries(split, split_rows, summaries_dir)

    if len(selected_splits) > 1:
        # Add gain columns before writing combined summary
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
                row.setdefault("gain_from_base_react", float(row["accuracy"]) - base_accuracy)
        write_combined_summary(all_rows, summaries_dir)

    # Clean up the single shared environment
    primary_env.close()

    print(f"\n{'='*60}")
    print("All done! Summaries written to:")
    print(f"  {summaries_dir}")
    print(f"{'='*60}")
    sys.stdout.flush()
    sys.stderr.flush()
    # WebShop's full catalog keeps millions of product/goal objects in memory;
    # normal interpreter teardown can burn minutes after outputs are written.
    os._exit(0)


if __name__ == "__main__":
    main()
