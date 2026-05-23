#!/usr/bin/env python3
"""
Run a small ScienceWorld framework suite.

Frameworks:
1. ReAct
2. ReAct + Context Retrieval (CR)
3. ReAct + Tool Retrieval (TR)
4. ReAct + CR + TR
5. ReAct + hard_neg CR + TR

This mirrors the WebShop/InterCode SQL suite pattern but keeps ScienceWorld's
task discovery, scoring, and prompt surface separate.
"""

import argparse
import csv
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FRAMEWORK_ORDER = [
    "react",
    "react_cr",
    "react_tr",
    "react_cr_tr",
    "react_hard_neg_cr_tr",
]

FRAMEWORK_DISPLAY = {
    "react": "ReAct",
    "react_reflexion": "React + Reflexion (final trials)",
    "react_cr": "ReAct + Context Retrieval (CR)",
    "react_tr": "ReAct + Tool Retrieval (TR)",
    "react_cr_tr": "ReAct + CR + TR",
    "react_hard_neg_cr_tr": "ReAct + hard_neg CR + TR",
}


@dataclass
class TaskInfo:
    split: str
    task_spec: str
    task_name: str
    variation_idx: int
    task_id_str: str


def parse_list_arg(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def ensure_clean_dir(path: Path, resume: bool = False) -> None:
    if resume:
        path.mkdir(parents=True, exist_ok=True)
        return
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_prompts(prompts_path: Path) -> Dict[str, str]:
    with open(prompts_path, "r") as f:
        return json.load(f)


def count_action_steps(history_items: List[Dict[str, str]]) -> int:
    return sum(1 for item in history_items if item.get("label") == "action")


def make_shared_scienceworld_env(jar_path: Optional[str], env_step_limit: int):
    try:
        from scienceworld import ScienceWorldEnv as _ScienceWorldEnv
    except ImportError as exc:
        raise ImportError(
            "Could not import ScienceWorld. Install with: pip install scienceworld. "
            "Java 1.8+ is also required."
        ) from exc
    return _ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)


def get_split_variations(env: Any, split: str) -> List[int]:
    if split == "train":
        return list(env.get_variations_train())
    if split == "dev":
        return list(env.get_variations_dev())
    if split == "test":
        return list(env.get_variations_test())
    raise ValueError(f"Unsupported ScienceWorld split: {split}")


def discover_tasks_for_split(
    env: Any,
    task_specs: Sequence[str],
    split: str,
    num_tasks: int,
    simplification_str: str,
    max_variations_per_task: int = 0,
) -> List[TaskInfo]:
    task_infos: List[TaskInfo] = []
    for task_spec in task_specs:
        env.load(task_spec, 0, simplification_str)
        task_name = getattr(env, "taskName", task_spec)
        variations = get_split_variations(env, split)

        for variation_count, variation_idx in enumerate(variations):
            if max_variations_per_task > 0 and variation_count >= max_variations_per_task:
                break
            task_infos.append(
                TaskInfo(
                    split=split,
                    task_spec=task_spec,
                    task_name=task_name,
                    variation_idx=int(variation_idx),
                    task_id_str=f"scienceworld_{task_name}_var_{variation_idx}",
                )
            )
            if len(task_infos) >= num_tasks:
                return task_infos
    return task_infos


def build_scienceworld_prompt(prompts: Dict[str, str]) -> str:
    prompt_parts = []
    for key in sorted(prompts.keys()):
        if key.startswith("scienceworld_react_"):
            prompt_parts.append(prompts[key])
    if prompt_parts:
        return "\n\n".join(prompt_parts)

    return (
        "You are an AI agent solving ScienceWorld text-game tasks.\n"
        "Use one concise action at a time. Prefer exact valid actions shown in the observation.\n"
        "Useful actions often include look around, inventory, examine, take, put, open, close, "
        "activate, deactivate, pour, mix, focus on, move, and teleport actions.\n"
        "Use think: for brief private reasoning when helpful.\n\n"
        "Example:\n"
        "Obs: Task: determine whether a material changes state.\n"
        "Action: look around\n"
        "Obs: You see the room and available objects.\n"
        "Action: inventory\n"
        "Obs: Your inventory is empty.\n"
    )


def make_env(
    task: TaskInfo,
    shared_env: Any,
    simplification_str: str,
    jar_path: Optional[str],
    env_step_limit: int,
    max_valid_actions: int,
):
    from src.envs.scienceworld_env import ScienceWorldEnv

    return ScienceWorldEnv(
        task_name=task.task_spec,
        variation_idx=task.variation_idx,
        simplification_str=simplification_str,
        jar_path=jar_path,
        env_step_limit=env_step_limit,
        max_valid_actions=max_valid_actions,
        scienceworld_env=shared_env,
    )


def extract_task_desc(observation: str) -> str:
    from src.envs.scienceworld_env import ScienceWorldEnv

    return ScienceWorldEnv.get_task_description(observation)


def compute_metrics(
    task_ids: Sequence[str],
    attempt_records: Sequence[Dict[str, Any]],
    reward_threshold: float = 1.0,
) -> Dict[str, Any]:
    total_tasks = len(task_ids)
    task_id_set = set(task_ids)
    success_task_ids = {
        row.get("task_id")
        for row in attempt_records
        if row.get("task_id") in task_id_set
        and float(row.get("reward", 0) or 0) >= reward_threshold
    }
    success_count = len(success_task_ids)
    avg_reward = mean(float(r.get("reward", 0) or 0) for r in attempt_records) if attempt_records else 0.0
    avg_steps = mean(float(r.get("step_num", 0) or 0) for r in attempt_records) if attempt_records else 0.0

    return {
        "success": int(success_count),
        "total": int(total_tasks),
        "success_total": f"{int(success_count)} / {int(total_tasks)}",
        "accuracy": float(success_count / total_tasks) if total_tasks else 0.0,
        "avg_reward": float(avg_reward),
        "avg_steps_per_trial": float(avg_steps),
        "avg_steps_per_task": float(avg_steps),
        "attempt_count": int(len(attempt_records)),
    }


def run_react_baseline(
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    shared_env: Any,
    simplification_str: str,
    jar_path: Optional[str],
    env_step_limit: int,
    max_valid_actions: int,
    reward_threshold: float,
    quiet: bool,
) -> Dict[str, Any]:
    from src.frameworks.react import ReAct

    agent = ReAct(model=model, to_print=not quiet)
    base_prompt = build_scienceworld_prompt(prompts)
    attempts: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    world_log = run_dir / "world.log"

    with open(world_log, "w") as wf:
        wf.write("ReAct baseline run (ScienceWorld)\n")

    for i, task in enumerate(task_infos):
        env = make_env(task, shared_env, simplification_str, jar_path, env_step_limit, max_valid_actions)
        success = False
        reward = 0.0
        step_num = 0
        task_desc = ""
        history_items: List[Dict[str, str]] = []
        try:
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob)
            history, _raw_success = agent.run(env=env, base_prompt=base_prompt, memory=[], start_ob=ob)
            history_items = history.to_json()
            step_num = count_action_steps(history_items)
            reward = env.last_score
            success = reward >= reward_threshold
        except Exception as exc:
            print(
                f"[react] Task failed with error ({task.task_id_str}): {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            env.close()

        attempt = {
            "task_id": task.task_id_str,
            "task_name": task.task_name,
            "variation_idx": task.variation_idx,
            "trial_num": 1,
            "step_num": step_num,
            "reward": reward,
            "success": success,
        }
        trajectory = {
            **attempt,
            "task_desc": task_desc,
            "steps": history_items,
            "split": task.split,
        }
        attempts.append(attempt)
        trajectories.append(trajectory)
        with open(world_log, "a") as wf:
            wf.write(
                f"Task #{i}: {task.task_id_str} - {'SUCCESS' if success else 'FAIL'} "
                f"(score={reward:.2f}, steps={step_num})\n"
            )

    save_json(run_dir / "attempts.json", attempts)
    save_json(run_dir / "trajectories.json", trajectories)
    return compute_metrics([t.task_id_str for t in task_infos], attempts, reward_threshold)


def run_memory_agent_variant(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    memory_bank_path: Path,
    shared_env: Any,
    simplification_str: str,
    jar_path: Optional[str],
    env_step_limit: int,
    max_valid_actions: int,
    reward_threshold: float,
    quiet: bool,
    max_learnings: int,
    min_valid_level: str,
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.hard_neg_memory_agent import HardNegMemoryAgent
    from src.frameworks.memory_retrieval_v2.agents.memory_agent import MemoryAgent

    class ContextOnlyAgent(MemoryAgent):
        def _get_help_instructions(self) -> str:
            return ""

        def _execute_help_tool(self, query: str) -> str:
            return "Help tool is disabled for this Context-Only run."

    if framework_id == "react_cr":
        agent = ContextOnlyAgent(model=model, to_print=not quiet, env_kind="scienceworld")
        include_context = True
    elif framework_id == "react_tr":
        agent = MemoryAgent(model=model, to_print=not quiet, env_kind="scienceworld")
        include_context = False
    elif framework_id == "react_cr_tr":
        agent = MemoryAgent(model=model, to_print=not quiet, env_kind="scienceworld")
        include_context = True
    elif framework_id == "react_hard_neg_cr_tr":
        agent = HardNegMemoryAgent(model=model, to_print=not quiet, env_kind="scienceworld")
        include_context = True
    else:
        raise ValueError(f"Unsupported memory variant: {framework_id}")

    base_prompt = build_scienceworld_prompt(prompts)
    attempts: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    world_log = run_dir / "world.log"

    with open(world_log, "w") as wf:
        wf.write(f"{framework_id} run (ScienceWorld)\n")
        wf.write(f"memory_bank={memory_bank_path}\n")

    for i, task in enumerate(task_infos):
        env = make_env(task, shared_env, simplification_str, jar_path, env_step_limit, max_valid_actions)
        success = False
        reward = 0.0
        step_num = 0
        task_desc = ""
        history_items: List[Dict[str, str]] = []
        try:
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob)
            history, _raw_success = agent.run(
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
            history_items = history.to_json()
            step_num = count_action_steps(history_items)
            reward = env.last_score
            success = reward >= reward_threshold
        except Exception as exc:
            print(
                f"[{framework_id}] Task failed with error ({task.task_id_str}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            env.close()

        attempt = {
            "task_id": task.task_id_str,
            "task_name": task.task_name,
            "variation_idx": task.variation_idx,
            "trial_num": 1,
            "step_num": step_num,
            "reward": reward,
            "success": success,
        }
        trajectory = {
            **attempt,
            "task_desc": task_desc,
            "steps": history_items,
            "split": task.split,
        }
        attempts.append(attempt)
        trajectories.append(trajectory)
        with open(world_log, "a") as wf:
            wf.write(
                f"Task #{i}: {task.task_id_str} - {'SUCCESS' if success else 'FAIL'} "
                f"(score={reward:.2f}, steps={step_num})\n"
            )

    save_json(run_dir / "attempts.json", attempts)
    save_json(run_dir / "trajectories.json", trajectories)
    metrics = compute_metrics([t.task_id_str for t in task_infos], attempts, reward_threshold)
    with open(world_log, "a") as wf:
        wf.write("-----\n")
        wf.write(f"SUCCESS: {metrics['success']}\n")
        wf.write(f"FAIL: {metrics['total'] - metrics['success']}\n")
        wf.write(f"TOTAL: {metrics['total']}\n")
        wf.write(f"ACCURACY: {metrics['accuracy']:.4f}\n")
        wf.write(f"AVG SCORE: {metrics['avg_reward']:.4f}\n")
        wf.write("-----\n")
    return metrics


def _latest_reflexion_for_trial(run_dir: Path, task_id: str, trial_num: int) -> str:
    reflexions_path = run_dir / "reflexions.json"
    if not reflexions_path.exists():
        return ""
    try:
        with open(reflexions_path, "r") as f:
            reflexions = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    for entry in reversed(reflexions):
        if entry.get("task_id") == task_id and int(entry.get("trial_num", 0) or 0) == int(trial_num):
            return entry.get("reflexion", "") or ""
    return ""


def run_reflexion(
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    shared_env: Any,
    simplification_str: str,
    jar_path: Optional[str],
    env_step_limit: int,
    max_valid_actions: int,
    reward_threshold: float,
    quiet: bool,
    max_trials: int,
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_allocation import MemoryAllocationReflexion

    agent = MemoryAllocationReflexion(
        model=model,
        to_print=not quiet,
        success_threshold=reward_threshold,
    )
    base_prompt = build_scienceworld_prompt(prompts)
    task_states: Dict[str, Dict[str, Any]] = {
        task.task_id_str: {
            "memory": [],
            "is_success": False,
            "final_attempt": None,
        }
        for task in task_infos
    }
    all_trajectories: List[Dict[str, Any]] = []
    world_log = run_dir / "world.log"

    with open(world_log, "w") as wf:
        wf.write("Reflexion run (ScienceWorld)\n")
        wf.write(f"max_trials={max_trials}, reward_threshold={reward_threshold}\n")

    for trial_num in range(1, max_trials + 1):
        pending_tasks = [t for t in task_infos if not task_states[t.task_id_str]["is_success"]]
        if not pending_tasks:
            break

        with open(world_log, "a") as wf:
            wf.write(f"Trial {trial_num}: {len(pending_tasks)} pending tasks\n")

        for i, task in enumerate(pending_tasks):
            state = task_states[task.task_id_str]
            env = make_env(task, shared_env, simplification_str, jar_path, env_step_limit, max_valid_actions)
            success = False
            reward = 0.0
            step_num = 0
            task_desc = ""
            history_items: List[Dict[str, str]] = []
            try:
                ob, _info = env.reset()
                task_desc = extract_task_desc(ob)
                history, _raw_success = agent.run(
                    env=env,
                    base_prompt=base_prompt,
                    memory=state["memory"],
                    start_ob=ob,
                    task_id=task.task_id_str,
                    trial_num=trial_num,
                    log_dir=str(run_dir),
                    task_desc=task_desc,
                )
                history_items = history.to_json()
                step_num = count_action_steps(history_items)
                reward = env.last_score
                success = reward >= reward_threshold
            except Exception as exc:
                print(
                    f"[react_reflexion] Task failed with error ({task.task_id_str}): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                env.close()

            attempt = {
                "task_id": task.task_id_str,
                "task_name": task.task_name,
                "variation_idx": task.variation_idx,
                "trial_num": trial_num,
                "step_num": step_num,
                "reward": reward,
                "success": success,
            }
            trajectory = {
                **attempt,
                "task_desc": task_desc,
                "steps": history_items,
                "split": task.split,
            }
            all_trajectories.append(trajectory)
            state["final_attempt"] = attempt
            state["is_success"] = success

            if not success:
                reflexion = _latest_reflexion_for_trial(run_dir, task.task_id_str, trial_num)
                if reflexion:
                    state["memory"].append(reflexion)

            with open(world_log, "a") as wf:
                wf.write(
                    f"Trial {trial_num} task #{i}: {task.task_id_str} - "
                    f"{'SUCCESS' if success else 'FAIL'} "
                    f"(score={reward:.2f}, steps={step_num})\n"
                )

    final_attempts: List[Dict[str, Any]] = []
    for task in task_infos:
        attempt = task_states[task.task_id_str].get("final_attempt")
        if attempt is None:
            attempt = {
                "task_id": task.task_id_str,
                "task_name": task.task_name,
                "variation_idx": task.variation_idx,
                "trial_num": 0,
                "step_num": 0,
                "reward": 0.0,
                "success": False,
            }
        final_attempts.append(attempt)

    save_json(run_dir / "attempts.json", final_attempts)
    save_json(run_dir / "trajectories.json", all_trajectories)
    return compute_metrics([t.task_id_str for t in task_infos], final_attempts, reward_threshold)


def run_framework(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    prompts: Dict[str, str],
    memory_bank_path: Path,
    shared_env: Any,
    simplification_str: str,
    jar_path: Optional[str],
    env_step_limit: int,
    max_valid_actions: int,
    reward_threshold: float,
    quiet: bool,
    max_learnings: int,
    min_valid_level: str,
    max_trials: int,
) -> Dict[str, Any]:
    if framework_id == "react":
        return run_react_baseline(
            task_infos,
            run_dir,
            model,
            prompts,
            shared_env,
            simplification_str,
            jar_path,
            env_step_limit,
            max_valid_actions,
            reward_threshold,
            quiet,
        )
    if framework_id == "react_reflexion":
        return run_reflexion(
            task_infos,
            run_dir,
            model,
            prompts,
            shared_env,
            simplification_str,
            jar_path,
            env_step_limit,
            max_valid_actions,
            reward_threshold,
            quiet,
            max_trials,
        )
    return run_memory_agent_variant(
        framework_id,
        task_infos,
        run_dir,
        model,
        prompts,
        memory_bank_path,
        shared_env,
        simplification_str,
        jar_path,
        env_step_limit,
        max_valid_actions,
        reward_threshold,
        quiet,
        max_learnings,
        min_valid_level,
    )


def write_split_summaries(split: str, rows: List[Dict[str, Any]], summaries_dir: Path) -> None:
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

    save_json(json_path, {"split": split, "rows": rows})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "success_total",
                "accuracy",
                "avg_score",
                "avg_steps_per_trial",
                "avg_steps_per_task",
                "gain_from_base_react",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": row["framework"],
                    "success_total": row["success_total"],
                    "accuracy": f"{row['accuracy']:.4f}",
                    "avg_score": f"{row.get('avg_reward', 0):.4f}",
                    "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                    "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                    "gain_from_base_react": f"{row['gain_from_base_react']:.4f}",
                }
            )

    with open(md_path, "w") as f:
        f.write(f"# ScienceWorld Summary ({split})\n\n")
        f.write("| Framework | Success / Total | Accuracy | Avg Score | Avg Steps/Trial | Avg Steps/Task | Gain from ReAct |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['framework']} | {row['success_total']} | {row['accuracy']:.4f} | "
                f"{row.get('avg_reward', 0):.4f} | {row['avg_steps_per_trial']:.4f} | "
                f"{row['avg_steps_per_task']:.4f} | {row['gain_from_base_react']:+.4f} |\n"
            )


def write_combined_summary(all_rows: List[Dict[str, Any]], summaries_dir: Path) -> None:
    save_json(summaries_dir / "summary_all.json", {"rows": all_rows})
    with open(summaries_dir / "summary_all.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "framework",
                "success_total",
                "accuracy",
                "avg_score",
                "avg_steps_per_trial",
                "avg_steps_per_task",
                "gain_from_base_react",
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
                    "avg_score": f"{row.get('avg_reward', 0):.4f}",
                    "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                    "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                    "gain_from_base_react": f"{row.get('gain_from_base_react', 0):.4f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ScienceWorld framework suite")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks per split")
    parser.add_argument("--splits", type=str, default="dev", help="Comma-separated splits: train,dev,test")
    parser.add_argument("--task-ids", type=str, default="1-1,1-2,1-3", help="Comma-separated ScienceWorld task IDs or names")
    parser.add_argument("--frameworks", type=str, default=",".join(FRAMEWORK_ORDER), help="Comma-separated framework IDs")
    parser.add_argument("--max-variations-per-task", type=int, default=0, help="Optional cap on variations selected from each ScienceWorld task spec")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Chat model name")
    parser.add_argument("--embedding-provider", type=str, default="", choices=["", "gemini", "openai"], help="Embedding provider for retrieval")
    parser.add_argument("--memory-bank", type=str, default="data/scienceworld/knowledge_base_seed.json", help="Path to memory bank JSON")
    parser.add_argument("--runs-root", type=str, default="scienceworld_runs/memory_retrieval_v2/memory_agent_runs", help="Root output directory")
    parser.add_argument("--prompts-path", type=str, default="data/scienceworld/prompts/scienceworld_prompts.json", help="ScienceWorld prompts JSON")
    parser.add_argument("--jar-path", type=str, default=None, help="Optional ScienceWorld JAR path")
    parser.add_argument("--simplifications", type=str, default="easy", help="ScienceWorld simplification string")
    parser.add_argument("--env-step-limit", type=int, default=100, help="ScienceWorld step limit")
    parser.add_argument("--max-valid-actions", type=int, default=80, help="Valid actions to include in each observation")
    parser.add_argument("--reward-threshold", type=float, default=1.0, help="Normalized score threshold for success")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-step framework print output")
    parser.add_argument("--resume", action="store_true", help="Preserve existing run dirs instead of wiping")
    parser.add_argument("--prepare-only", action="store_true", help="Discover tasks and write config without running")
    parser.add_argument("--max-learnings", type=int, default=10, help="Max learnings for CR")
    parser.add_argument("--max-trials", type=int, default=7, help="Max trials for Reflexion runs")
    parser.add_argument(
        "--min-valid-level",
        type=str,
        default="VALID_SAME_TRIAL",
        choices=["", "CANDIDATE", "VALID_SAME_TRIAL", "VALID_NEXT_TRIAL"],
        help="Minimum validation level for retrieved learnings",
    )
    parser.add_argument("--seed", type=int, default=0, help="Harness seed")
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass

    embedding_provider = args.embedding_provider or "gemini"
    os.environ["LTM_EMBEDDING_PROVIDER"] = embedding_provider
    print(f"Using model={args.model}, embedding_provider={embedding_provider}, seed={args.seed}")

    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = (base_dir / args.prompts_path).resolve()
    runs_root = (base_dir / args.runs_root).resolve()
    memory_bank_path = (base_dir / args.memory_bank).resolve()
    summaries_dir = runs_root / "summaries"
    selected_splits = parse_list_arg(args.splits)
    selected_frameworks = parse_list_arg(args.frameworks)
    task_specs = parse_list_arg(args.task_ids)

    for framework_id in selected_frameworks:
        if framework_id not in FRAMEWORK_DISPLAY:
            raise ValueError(f"Unsupported framework: {framework_id}")

    memory_frameworks = {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}
    if memory_frameworks & set(selected_frameworks) and not memory_bank_path.exists():
        raise FileNotFoundError(f"Memory bank not found: {memory_bank_path}")

    prompts = load_prompts(prompts_path) if prompts_path.exists() else {}
    runs_root.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    shared_env = make_shared_scienceworld_env(args.jar_path, args.env_step_limit)
    try:
        all_split_tasks: Dict[str, List[TaskInfo]] = {}
        for split in selected_splits:
            tasks = discover_tasks_for_split(
                shared_env,
                task_specs,
                split,
                args.num_tasks,
                args.simplifications,
                args.max_variations_per_task,
            )
            all_split_tasks[split] = tasks
            print(f"Split '{split}': {len(tasks)} tasks")

        seed_tag = f"seed_{args.seed}"
        save_json(
            runs_root / "suite_config.json",
            {
                "created_at": datetime.now().isoformat(),
                "num_tasks": args.num_tasks,
                "splits": selected_splits,
                "task_ids": task_specs,
                "max_variations_per_task": args.max_variations_per_task,
                "frameworks": selected_frameworks,
                "model": args.model,
                "embedding_provider": embedding_provider,
                "memory_bank": str(memory_bank_path),
                "simplifications": args.simplifications,
                "env_step_limit": args.env_step_limit,
                "max_valid_actions": args.max_valid_actions,
                "reward_threshold": args.reward_threshold,
                "max_trials": args.max_trials,
                "max_learnings": args.max_learnings,
                "min_valid_level": args.min_valid_level,
                "seed": args.seed,
                "prepare_only": args.prepare_only,
            },
        )

        if args.prepare_only:
            print("Prepare-only mode. Exiting.")
            return

        all_rows: List[Dict[str, Any]] = []
        for split in selected_splits:
            task_infos = all_split_tasks[split]
            split_rows: List[Dict[str, Any]] = []
            for framework_id in selected_frameworks:
                run_dir = runs_root / split / framework_id / seed_tag
                ensure_clean_dir(run_dir, resume=args.resume)
                print(
                    f"\nRunning split={split} framework={framework_id} "
                    f"tasks={len(task_infos)} -> {run_dir}"
                )
                metrics = run_framework(
                    framework_id=framework_id,
                    task_infos=task_infos,
                    run_dir=run_dir,
                    model=args.model,
                    prompts=prompts,
                    memory_bank_path=memory_bank_path,
                    shared_env=shared_env,
                    simplification_str=args.simplifications,
                    jar_path=args.jar_path,
                    env_step_limit=args.env_step_limit,
                    max_valid_actions=args.max_valid_actions,
                    reward_threshold=args.reward_threshold,
                    quiet=args.quiet,
                    max_learnings=args.max_learnings,
                    min_valid_level=args.min_valid_level,
                    max_trials=args.max_trials,
                )
                save_json(run_dir / "metrics.json", metrics)
                row = {
                    "split": split,
                    "framework_id": framework_id,
                    "framework": FRAMEWORK_DISPLAY[framework_id],
                    "run_dir": str(run_dir),
                    **metrics,
                }
                split_rows.append(row)
                all_rows.append(row)
                print(
                    f"  Results: {metrics['success_total']} success, "
                    f"accuracy={metrics['accuracy']:.4f}, avg_score={metrics['avg_reward']:.4f}"
                )
            write_split_summaries(split, split_rows, summaries_dir)

        if len(selected_splits) > 1:
            rows_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in all_rows:
                rows_by_split[row["split"]].append(row)
            for rows in rows_by_split.values():
                base_accuracy = next((float(r["accuracy"]) for r in rows if r["framework_id"] == "react"), 0.0)
                for row in rows:
                    row.setdefault("gain_from_base_react", float(row["accuracy"]) - base_accuracy)
            write_combined_summary(all_rows, summaries_dir)

        print(f"\nSuite run complete. Summaries written to: {summaries_dir}")
    finally:
        shared_env.close()


if __name__ == "__main__":
    main()
