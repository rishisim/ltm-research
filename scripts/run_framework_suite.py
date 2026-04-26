#!/usr/bin/env python3
"""
Run a standardized ALFWorld framework suite on alfworld_mini seen/unseen splits.

Frameworks:
1. ReAct
2. ReAct + Reflexion (max N trials)
3. ReAct + Context Retrieval (CR)
4. ReAct + Tool Retrieval (TR)
5. ReAct + CR + TR
6. ReAct + hard_neg CR + TR

The script also supports:
- archiving existing memory_agent_runs contents to misc/old_runs
- generating deterministic task manifests
- writing per-framework metrics and split summaries
"""

import argparse
import copy
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

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

SPLIT_ALIAS = {
    "valid_seen": "seen",
    "valid_unseen": "unseen",
}


@dataclass
class TaskInfo:
    split: str
    task_name: str
    trial_name: str
    task_id: str
    task_dir: Path


def parse_list_arg(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def discover_tasks_for_split(base_dir: Path, split: str, num_tasks: int) -> List[str]:
    """Enumerate every (task, trial) pair so all game environments are covered.

    Each task directory may contain multiple trial sub-directories, each with
    its own game.tw-pddl. We include all valid trials so the full game count
    matches the official ALFWorld split sizes (e.g. 134 for valid_unseen).
    """
    split_dir = base_dir / "alfworld_mini" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    specs: List[str] = []

    for task_name in sorted(os.listdir(split_dir)):
        task_dir = split_dir / task_name
        if not task_dir.is_dir():
            continue

        trial_dirs = sorted(
            d for d in os.listdir(task_dir) if (task_dir / d).is_dir()
        )

        for trial_name in trial_dirs:
            game_file = task_dir / trial_name / "game.tw-pddl"
            if not game_file.exists():
                continue
            specs.append(f"{split}:{task_name}/{trial_name}")
            if len(specs) >= num_tasks:
                return specs

    return specs


def task_infos_from_specs(base_dir: Path, specs: Sequence[str]) -> List[TaskInfo]:
    task_infos: List[TaskInfo] = []

    for spec in specs:
        if ":" not in spec:
            continue
        split, task_path = spec.split(":", 1)
        if "/" not in task_path:
            continue
        task_name, trial_name = task_path.rsplit("/", 1)
        task_dir = base_dir / "alfworld_mini" / split / task_name / trial_name
        game_file = task_dir / "game.tw-pddl"
        if not game_file.exists():
            continue
        task_infos.append(
            TaskInfo(
                split=split,
                task_name=task_name,
                trial_name=trial_name,
                task_id=f"{task_name}/{trial_name}",
                task_dir=task_dir,
            )
        )

    return task_infos


def choose_prompt_key(task_id: str) -> str:
    short_task_name = task_id.split("-")[0]
    if "pick_cool" in short_task_name:
        return "react_cool_0"
    if "pick_heat" in short_task_name:
        return "react_heat_0"
    if "pick_clean" in short_task_name:
        return "react_clean_0"
    if "pick_two" in short_task_name:
        return "react_puttwo_0"
    if "look_at" in short_task_name:
        return "react_examine_0"
    return "react_put_0"


def extract_task_desc(observation: str) -> str:
    if "Your task is to:" in observation:
        return observation.split("Your task is to:")[-1].strip().split("\n")[0].strip()
    for line in observation.split("\n"):
        if line.strip():
            return line.strip()
    return ""


def count_action_steps(history_items: List[Dict[str, str]]) -> int:
    return sum(1 for item in history_items if item.get("label") == "action")


def make_env(config_template: Dict[str, Any], task_dir: Path) -> Any:
    from src.envs.alfworld_env import AlfworldEnv

    config = copy.deepcopy(config_template)
    config["dataset"]["eval_ood_data_path"] = str(task_dir)
    config["general"]["evaluate"]["batch_size"] = 1
    return AlfworldEnv(config, split="eval_out_of_distribution")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(
    task_ids: Sequence[str],
    attempt_records: Sequence[Dict[str, Any]],
    final_success_override: Optional[int] = None,
) -> Dict[str, Any]:
    total_tasks = len(task_ids)
    success_count = (
        final_success_override
        if final_success_override is not None
        else sum(1 for r in attempt_records if r.get("success"))
    )

    accuracy = (success_count / total_tasks) if total_tasks else 0.0
    avg_steps_per_trial = (
        mean(float(r.get("step_num", 0)) for r in attempt_records) if attempt_records else 0.0
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
    config_template: Dict[str, Any],
    prompts: Dict[str, str],
    quiet: bool,
) -> Dict[str, Any]:
    from src.frameworks.react import ReAct

    agent = ReAct(model=model, to_print=not quiet)
    attempts: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    world_log = run_dir / "world.log"

    with open(world_log, "w") as wf:
        wf.write("ReAct baseline run\n")

    for i, task in enumerate(task_infos):
        success = False
        step_num = 0
        history_items: List[Dict[str, str]] = []

        env = make_env(config_template, task.task_dir)
        try:
            ob, _info = env.reset()
            prompt_key = choose_prompt_key(task.task_id)
            base_prompt = prompts.get(prompt_key, prompts.get("react_put_0", ""))
            history, success = agent.run(
                env=env,
                base_prompt=base_prompt,
                memory=[],
                start_ob=ob,
            )
            history_items = history.to_json()
            step_num = count_action_steps(history_items)
        except Exception as e:
            if not quiet:
                print(f"[react] Task failed with error ({task.task_id}): {e}")
            success = False
            step_num = 0
            history_items = []
        finally:
            env.close()

        attempts.append(
            {
                "task_id": task.task_id,
                "trial_num": 1,
                "step_num": step_num,
                "success": success,
            }
        )
        trajectories.append(
            {
                "task_id": task.task_id,
                "task_type": task.task_name.split("-")[0],
                "task_desc": "",
                "trial_num": 1,
                "steps": history_items,
                "success": success,
                "step_num": step_num,
                "split": task.split,
            }
        )

        with open(world_log, "a") as wf:
            wf.write(f"Task #{i}: {task.task_id} - {'SUCCESS' if success else 'FAIL'}\n")

    with open(run_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    metrics = compute_metrics([t.task_id for t in task_infos], attempts)
    return metrics


def run_memory_agent_variant(
    framework_id: str,
    task_infos: Sequence[TaskInfo],
    run_dir: Path,
    model: str,
    config_template: Dict[str, Any],
    prompts: Dict[str, str],
    memory_bank_path: Path,
    quiet: bool,
    max_learnings: int = 25,
    min_valid_level: str = "",
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_agent import MemoryAgent
    from src.frameworks.memory_retrieval_v2.agents.hard_neg_memory_agent import HardNegMemoryAgent

    class ContextOnlyAgent(MemoryAgent):
        def _get_help_instructions(self) -> str:
            return ""

        def _execute_help_tool(self, query: str) -> str:
            return "Help tool is disabled for this Context-Only run."

    if framework_id == "react_cr":
        agent = ContextOnlyAgent(model=model, to_print=not quiet)
        include_context = True
        use_hard_neg = False
    elif framework_id == "react_tr":
        agent = MemoryAgent(model=model, to_print=not quiet)
        include_context = False
        use_hard_neg = False
    elif framework_id == "react_cr_tr":
        agent = MemoryAgent(model=model, to_print=not quiet)
        include_context = True
        use_hard_neg = False
    elif framework_id == "react_hard_neg_cr_tr":
        agent = HardNegMemoryAgent(model=model, to_print=not quiet)
        include_context = True
        use_hard_neg = True
    else:
        raise ValueError(f"Unsupported memory variant: {framework_id}")

    attempts: List[Dict[str, Any]] = []
    world_log = run_dir / "world.log"

    with open(world_log, "w") as wf:
        wf.write(f"{framework_id} run\n")
        wf.write(f"memory_bank={memory_bank_path}\n")
        wf.write(f"hard_negative={use_hard_neg}\n")

    for i, task in enumerate(task_infos):
        success = False
        step_num = 0
        env = make_env(config_template, task.task_dir)
        try:
            ob, _info = env.reset()
            task_desc = extract_task_desc(ob)
            prompt_key = choose_prompt_key(task.task_id)
            base_prompt = prompts.get(prompt_key, prompts.get("react_put_0", ""))

            history, success = agent.run(
                env=env,
                base_prompt=base_prompt,
                memory=[],
                start_ob=ob,
                task_id=task.task_id,
                trial_num=1,
                log_dir=str(run_dir),
                task_desc=(task_desc if include_context else ""),
                memory_bank_path=str(memory_bank_path),
                max_learnings=max_learnings,
                min_valid_level=min_valid_level,
            )
            step_num = count_action_steps(history.to_json())
        except Exception as e:
            if not quiet:
                print(f"[{framework_id}] Task failed with error ({task.task_id}): {e}")
            success = False
            step_num = 0
        finally:
            env.close()

        attempts.append(
            {
                "task_id": task.task_id,
                "trial_num": 1,
                "step_num": step_num,
                "success": success,
            }
        )
        with open(world_log, "a") as wf:
            wf.write(f"Task #{i}: {task.task_id} - {'SUCCESS' if success else 'FAIL'}\n")

    metrics = compute_metrics([t.task_id for t in task_infos], attempts)
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
    config_template: Dict[str, Any],
    prompts: Dict[str, str],
    quiet: bool,
) -> Dict[str, Any]:
    from src.frameworks.memory_retrieval_v2.agents.memory_allocation import MemoryAllocationReflexion

    agent = MemoryAllocationReflexion(model=model, to_print=not quiet)
    world_log = run_dir / "world.log"

    state: Dict[str, Dict[str, Any]] = {
        task.task_id: {"memory": [], "is_success": False, "skip": False}
        for task in task_infos
    }

    with open(world_log, "w") as wf:
        wf.write("Reflexion run\n")
        wf.write(f"max_trials={max_trials}\n")

    trials_executed = 0

    for trial_idx in range(max_trials):
        trials_executed += 1
        trial_successes = 0
        trial_failures = 0
        additional_successes = 0

        with open(world_log, "a") as wf:
            wf.write(f"\n\n***** Start Trial #{trial_idx} *****\n\n")

        for i, task in enumerate(task_infos):
            task_state = state[task.task_id]
            if task_state["is_success"] or task_state["skip"]:
                trial_successes += 1
                with open(world_log, "a") as wf:
                    wf.write(f"Task #{i} Trial #{trial_idx}: SUCCESS (previous)\n")
                continue

            success = False
            env = make_env(config_template, task.task_dir)
            try:
                ob, _info = env.reset()
                task_desc = extract_task_desc(ob)
                prompt_key = choose_prompt_key(task.task_id)
                base_prompt = prompts.get(prompt_key, prompts.get("react_put_0", ""))

                _history, success = agent.run(
                    env=env,
                    base_prompt=base_prompt,
                    memory=task_state["memory"],
                    start_ob=ob,
                    task_id=task.task_id,
                    trial_num=trial_idx + 1,
                    log_dir=str(run_dir),
                    task_desc=task_desc,
                )
            except Exception as e:
                if not quiet:
                    print(f"[react_reflexion] Task failed with error ({task.task_id}): {e}")
                task_state["skip"] = True
                success = False
            finally:
                env.close()

            task_state["is_success"] = task_state["is_success"] or success
            if success:
                trial_successes += 1
                additional_successes += 1
            else:
                trial_failures += 1

            with open(world_log, "a") as wf:
                wf.write(
                    f"Task #{i} Trial #{trial_idx}: {'SUCCESS' if success else 'FAIL'}\n"
                )

        reflexions_path = run_dir / "reflexions.json"
        if reflexions_path.exists():
            with open(reflexions_path, "r") as f:
                reflexions = json.load(f)

            for task in task_infos:
                task_state = state[task.task_id]
                if task_state["is_success"] or task_state["skip"]:
                    continue
                for entry in reversed(reflexions):
                    if (
                        entry.get("task_id") == task.task_id
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
            wf.write(f"ADDITIONAL SUCCESS: {additional_successes}\n")
            wf.write(f"FAIL: {trial_failures}\n")
            wf.write(f"TOTAL: {len(task_infos)}\n")
            wf.write(f"ACCURACY: {accuracy:.2f}\n")
            wf.write("-----\n")
            wf.write(f"\n\n***** End Trial #{trial_idx} *****\n\n")

        if all(item["is_success"] or item["skip"] for item in state.values()):
            break

    trajectories_path = run_dir / "trajectories.json"
    attempt_records: List[Dict[str, Any]] = []
    if trajectories_path.exists():
        with open(trajectories_path, "r") as f:
            trajectories = json.load(f)
        for entry in trajectories:
            attempt_records.append(
                {
                    "task_id": entry.get("task_id"),
                    "trial_num": int(entry.get("trial_num", 0)),
                    "step_num": int(entry.get("step_num", 0)),
                    "success": bool(entry.get("success", False)),
                }
            )

    final_success = sum(1 for item in state.values() if item["is_success"])
    metrics = compute_metrics(
        [t.task_id for t in task_infos],
        attempt_records,
        final_success_override=final_success,
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
                    "avg_steps_per_trial": f"{row['avg_steps_per_trial']:.4f}",
                    "avg_steps_per_task": f"{row['avg_steps_per_task']:.4f}",
                    "gain_from_base_react": f"{row['gain_from_base_react']:.4f}",
                }
            )

    with open(md_path, "w") as f:
        f.write(f"# Summary ({split})\n\n")
        f.write("| Framework | Success / Total | Accuracy | Avg Steps Per Trial | Avg Steps per Task | Gain from Base ReAct |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['framework']} | {row['success_total']} | {row['accuracy']:.4f} | "
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
                "split",
                "framework",
                "success_total",
                "accuracy",
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
    config_template: Dict[str, Any],
    prompts: Dict[str, str],
    memory_bank_path: Path,
    quiet: bool,
) -> Dict[str, Any]:
    if framework_id == "react":
        return run_react_baseline(
            task_infos=task_infos,
            run_dir=run_dir,
            model=model,
            config_template=config_template,
            prompts=prompts,
            quiet=quiet,
        )
    if framework_id == "react_reflexion":
        return run_reflexion(
            task_infos=task_infos,
            run_dir=run_dir,
            model=model,
            max_trials=max_trials,
            config_template=config_template,
            prompts=prompts,
            quiet=quiet,
        )
    if framework_id in {"react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"}:
        return run_memory_agent_variant(
            framework_id=framework_id,
            task_infos=task_infos,
            run_dir=run_dir,
            model=model,
            config_template=config_template,
            prompts=prompts,
            memory_bank_path=memory_bank_path,
            quiet=quiet,
        )
    raise ValueError(f"Unknown framework id: {framework_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALFWorld framework suite")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of tasks per split (default: 20)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="valid_seen,valid_unseen",
        help="Comma-separated splits (default: valid_seen,valid_unseen)",
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
        default="alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json",
        help="Path to memory bank JSON",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="alfworld_runs/memory_retrieval_v2/memory_agent_runs",
        help="Root output directory",
    )
    parser.add_argument(
        "--misc-archive-root",
        type=str,
        default="alfworld_runs/memory_retrieval_v2/misc/old_runs/memory_agent_runs",
        help="Archive directory root for old runs",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Archive current runs-root contents before preparing/running",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare manifests/directories, do not execute framework runs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-step framework print output",
    )
    args = parser.parse_args()

    # Set embedding provider before any retrieval module is imported.
    # Default: ALWAYS use Gemini embeddings, regardless of chat model. The
    # knowledge bases were extracted with Gemini-embeddings (commit 94b2b8f),
    # and retrieval must use the same embedding model to stay consistent with
    # the KB. Mixing chat models for the cross-lab study is fine; mixing
    # embedding models silently changes the retrieval distribution and
    # confounds the comparison. Override via --embedding-provider only if you
    # know what you're doing (e.g., a fresh KB rebuild).
    def _auto_embedding_provider(model: str) -> str:
        return "gemini"

    embedding_provider = args.embedding_provider if args.embedding_provider else _auto_embedding_provider(args.model)
    os.environ["LTM_EMBEDDING_PROVIDER"] = embedding_provider
    print(f"Using model={args.model}, embedding_provider={embedding_provider}")

    base_dir = Path(__file__).resolve().parents[1]
    config_path = base_dir / "data" / "alfworld" / "base_config.yaml"
    prompts_path = base_dir / "data" / "alfworld" / "prompts" / "alfworld_3prompts.json"
    runs_root = (base_dir / args.runs_root).resolve()
    misc_archive_root = (base_dir / args.misc_archive_root).resolve()
    memory_bank_path = (base_dir / args.memory_bank).resolve()

    if not memory_bank_path.exists():
        raise FileNotFoundError(f"Memory bank not found: {memory_bank_path}")

    selected_splits = parse_list_arg(args.splits)
    selected_frameworks = parse_list_arg(args.frameworks)

    for split in selected_splits:
        if split not in SPLIT_ALIAS:
            raise ValueError(f"Unsupported split: {split}")
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
    manifests_dir = runs_root / "manifests"
    summaries_dir = runs_root / "summaries"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Prepare deterministic task manifests
    manifests: Dict[str, Dict[str, Any]] = {}
    for split in selected_splits:
        task_specs = discover_tasks_for_split(base_dir, split, args.num_tasks)
        manifest = {
            "split": split,
            "num_tasks_requested": args.num_tasks,
            "num_tasks_selected": len(task_specs),
            "tasks": task_specs,
        }
        manifests[split] = manifest
        save_json(manifests_dir / f"{split}_{args.num_tasks}_tasks.json", manifest)

    # Persist top-level suite config
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
            "prepare_only": args.prepare_only,
        },
    )

    # Prepare split/framework directory skeleton ahead of execution.
    for split in selected_splits:
        split_alias = SPLIT_ALIAS[split]
        for framework_id in selected_frameworks:
            (runs_root / split_alias / framework_id).mkdir(parents=True, exist_ok=True)

    if args.prepare_only:
        print("Prepared manifests and suite config only. No framework runs executed.")
        return

    config_template = load_config(config_path)
    prompts = load_prompts(prompts_path)
    os.environ["ALFWORLD_DATA"] = str(base_dir / "data")

    all_rows: List[Dict[str, Any]] = []

    for split in selected_splits:
        split_alias = SPLIT_ALIAS[split]
        task_infos = task_infos_from_specs(base_dir, manifests[split]["tasks"])
        if not task_infos:
            print(f"No valid tasks for split={split}; skipping.")
            continue

        split_rows: List[Dict[str, Any]] = []

        for framework_id in selected_frameworks:
            run_dir = runs_root / split_alias / framework_id
            ensure_clean_dir(run_dir)

            print(
                f"Running split={split} framework={framework_id} tasks={len(task_infos)} "
                f"-> {run_dir}"
            )
            metrics = run_framework(
                framework_id=framework_id,
                task_infos=task_infos,
                run_dir=run_dir,
                model=args.model,
                max_trials=args.max_trials,
                config_template=config_template,
                prompts=prompts,
                memory_bank_path=memory_bank_path,
                quiet=args.quiet,
            )

            row = {
                "split": split,
                "framework_id": framework_id,
                "framework": (
                    "React + Reflexion (final trials, max 7)"
                    if framework_id == "react_reflexion"
                    else FRAMEWORK_DISPLAY[framework_id]
                ),
                "success_total": metrics["success_total"],
                "accuracy": metrics["accuracy"],
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

    # Include gain columns in combined summary as well
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
    print(f"Suite run complete. Summaries written to: {summaries_dir}")


if __name__ == "__main__":
    main()
