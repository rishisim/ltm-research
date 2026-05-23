#!/usr/bin/env python3
"""Prepare, build, and audit the final ALFWorld train-only KB.

This script is intentionally scoped to the final ALFWorld KB campaign. It
avoids the legacy memory-allocation runner because that runner does not record
all final provenance fields and previous artifacts mixed trial depths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_ID = "alfworld_train_reflexion_trials3"
DEFAULT_FINAL_DIR = Path("final_runs/kb/alfworld_train_reflexion_trials3")
DEFAULT_MANIFEST = Path("final_runs/manifests/alfworld_train_reflexion_trials3.json")
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_PROVIDER = "gemini"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EXPECTED_TASKS = 120
DEFAULT_MAX_TRIALS = 3
SUCCESS_THRESHOLD = 1.0
VALID_LEVELS = {"VALID_SAME_TRIAL", "VALID_NEXT_TRIAL", "CANDIDATE"}


PROMPT_KEYS = (
    ("pick_cool", "react_cool_0"),
    ("pick_heat", "react_heat_0"),
    ("pick_clean", "react_clean_0"),
    ("pick_two", "react_puttwo_0"),
    ("look_at", "react_examine_0"),
    ("pick_and_place", "react_put_0"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return REPO_ROOT


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    tmp.replace(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def command_line() -> str:
    return " ".join([Path(sys.executable).name] + [str(x) for x in sys.argv])


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def candidate_data_roots(explicit: Optional[str]) -> List[Path]:
    roots: List[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    if os.environ.get("ALFWORLD_DATA"):
        roots.append(Path(os.environ["ALFWORLD_DATA"]).expanduser())
    root = repo_root()
    roots.extend(
        [
            root / "alfworld_mini",
            root / "data",
            root / "data" / "alfworld",
        ]
    )
    deduped: List[Path] = []
    seen = set()
    for p in roots:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def split_dir_for(data_root: Path, split: str) -> Optional[Path]:
    candidates = [
        data_root / "json_2.1.1" / split,
        data_root / split,
        data_root / "alfworld_mini" / split,
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def discover_split(data_root: Path, split: str) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    split_dir = split_dir_for(data_root, split)
    if split_dir is None:
        return None, []

    tasks: List[Dict[str, Any]] = []
    for game_file in sorted(split_dir.rglob("game.tw-pddl")):
        task_path = game_file.parent
        task_id = task_path.relative_to(split_dir).as_posix()
        base_name = Path(task_id).parts[0] if Path(task_id).parts else task_id
        task_type = base_name.split("-")[0] if "-" in base_name else base_name
        tasks.append(
            {
                "task_id": task_id,
                "split": split,
                "task_type": task_type,
                "task_path": str(task_path),
                "game_file": str(game_file),
            }
        )
    return split_dir, tasks


def find_data_root(explicit: Optional[str]) -> Tuple[Optional[Path], Dict[str, Any]]:
    attempts = []
    for root in candidate_data_roots(explicit):
        split_dir, tasks = discover_split(root, "train")
        attempts.append(
            {
                "candidate": str(root),
                "exists": root.exists(),
                "train_split_dir": str(split_dir) if split_dir else None,
                "train_game_count": len(tasks),
            }
        )
        if split_dir and tasks:
            return root, {"attempts": attempts}
    return None, {"attempts": attempts}


def check_environment(args: argparse.Namespace) -> Dict[str, Any]:
    data_root, data_root_probe = find_data_root(args.alfworld_data)
    env_vars = {
        "ALFWORLD_DATA": bool(os.environ.get("ALFWORLD_DATA")),
        "LTM_OPENROUTER_API_KEY": bool(os.environ.get("LTM_OPENROUTER_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
    }

    imports = {
        "alfworld": module_available("alfworld"),
        "yaml": module_available("yaml"),
        "google.genai": module_available("google.genai"),
        "google.generativeai": module_available("google.generativeai"),
    }

    split_dirs: Dict[str, Optional[str]] = {}
    split_counts: Dict[str, int] = {}
    split_task_ids: Dict[str, List[str]] = {}
    train_tasks: List[Dict[str, Any]] = []

    if data_root:
        for split in ("train", "valid_seen", "valid_unseen"):
            split_dir, tasks = discover_split(data_root, split)
            split_dirs[split] = str(split_dir) if split_dir else None
            split_counts[split] = len(tasks)
            split_task_ids[split] = [t["task_id"] for t in tasks]
            if split == "train":
                train_tasks = tasks
    else:
        split_dirs = {"train": None, "valid_seen": None, "valid_unseen": None}
        split_counts = {"train": 0, "valid_seen": 0, "valid_unseen": 0}
        split_task_ids = {"train": [], "valid_seen": [], "valid_unseen": []}

    train_set = set(split_task_ids["train"])
    seen_set = set(split_task_ids["valid_seen"])
    unseen_set = set(split_task_ids["valid_unseen"])
    overlaps = {
        "train_valid_seen": sorted(train_set & seen_set),
        "train_valid_unseen": sorted(train_set & unseen_set),
        "valid_seen_valid_unseen": sorted(seen_set & unseen_set),
    }

    blockers = []
    warnings = []
    if not imports["alfworld"]:
        blockers.append("Python module 'alfworld' is not importable.")
    if data_root is None:
        blockers.append(
            "No ALFWorld task data root found. Set ALFWORLD_DATA or pass --alfworld-data to a root containing json_2.1.1/train or train/."
        )
    elif split_counts["train"] != args.expected_tasks:
        blockers.append(
            f"Train split discovery found {split_counts['train']} game files, expected exactly {args.expected_tasks}."
        )
    if data_root is not None and not split_dirs.get("valid_seen"):
        blockers.append("valid_seen split directory is missing, so train/eval overlap cannot be audited.")
    if data_root is not None and not split_dirs.get("valid_unseen"):
        blockers.append("valid_unseen split directory is missing, so train/eval overlap cannot be audited.")
    if overlaps["train_valid_seen"]:
        blockers.append(f"Train/valid_seen overlap has {len(overlaps['train_valid_seen'])} task IDs.")
    if overlaps["train_valid_unseen"]:
        blockers.append(f"Train/valid_unseen overlap has {len(overlaps['train_valid_unseen'])} task IDs.")
    if args.model.startswith("gemini") and not env_vars["LTM_OPENROUTER_API_KEY"]:
        blockers.append("LTM_OPENROUTER_API_KEY is missing for gemini chat calls routed through OpenRouter.")
    if args.embedding_provider == "gemini" and not env_vars["GEMINI_API_KEY"]:
        blockers.append("GEMINI_API_KEY is missing for gemini-embedding-001 cache construction.")
    if args.embedding_provider != "gemini":
        blockers.append(f"Embedding provider must be gemini for the final KB, got {args.embedding_provider!r}.")
    if args.max_trials != DEFAULT_MAX_TRIALS:
        blockers.append(f"max_trials must be {DEFAULT_MAX_TRIALS}, got {args.max_trials}.")

    if env_vars["GOOGLE_API_KEY"] and not env_vars["GEMINI_API_KEY"]:
        warnings.append("GOOGLE_API_KEY is set, but this repo's Gemini embedding path reads GEMINI_API_KEY.")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "imports": imports,
        "env_vars": env_vars,
        "data_root": str(data_root) if data_root else None,
        "data_root_probe": data_root_probe,
        "split_dirs": split_dirs,
        "split_counts": split_counts,
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "overlap_examples": {k: v[:10] for k, v in overlaps.items()},
        "train_tasks": train_tasks,
    }


def estimate_scope(task_count: int, max_trials: int) -> Dict[str, Any]:
    episodes = task_count * max_trials
    max_action_calls = episodes * 49
    max_reflexion_calls = episodes
    max_kb_extraction_calls = task_count
    return {
        "task_count": task_count,
        "max_trials": max_trials,
        "max_agent_episodes": episodes,
        "max_agent_action_llm_calls": max_action_calls,
        "max_reflexion_llm_calls": max_reflexion_calls,
        "kb_extraction_llm_calls": max_kb_extraction_calls,
        "estimated_total_llm_calls_upper_bound": max_action_calls + max_reflexion_calls + max_kb_extraction_calls,
        "runtime_note": (
            "Upper bound is conservative because tasks stop after success/done. "
            "At 2-5 seconds per action call, 360 episodes can plausibly take 10-25 hours before KB extraction."
        ),
    }


def build_manifest(args: argparse.Namespace, prep: Dict[str, Any], status: str, safe_for_eval: bool) -> Dict[str, Any]:
    train_tasks = prep.get("train_tasks", [])
    return {
        "artifact_id": ARTIFACT_ID,
        "status": status,
        "safe_for_eval": safe_for_eval,
        "generated_at_utc": utc_now(),
        "git_commit": git_commit(),
        "command": command_line(),
        "final_kb_dir": rel(Path(args.final_dir)),
        "manifest_path": rel(Path(args.manifest)),
        "config": {
            "model": args.model,
            "embedding_provider": args.embedding_provider,
            "embedding_model": DEFAULT_EMBEDDING_MODEL if args.embedding_provider == "gemini" else None,
            "seed": args.seed,
            "split": "train",
            "framework": "react_reflexion",
            "framework_list": ["react_reflexion"],
            "max_trials": args.max_trials,
            "max_learnings": "N/A_KB_construction_eval_only",
            "min_valid_level": "N/A_KB_construction_eval_only",
            "success_threshold": SUCCESS_THRESHOLD,
        },
        "prepare": {
            "ready": prep["ready"],
            "blockers": prep["blockers"],
            "warnings": prep["warnings"],
            "imports": prep["imports"],
            "env_vars_present": prep["env_vars"],
            "data_root": prep["data_root"],
            "split_dirs": prep["split_dirs"],
            "split_counts": prep["split_counts"],
            "overlap_counts": prep["overlap_counts"],
            "overlap_examples": prep["overlap_examples"],
        },
        "scope_estimate": estimate_scope(len(train_tasks) or args.expected_tasks, args.max_trials),
        "train_manifest": {
            "expected_task_count": args.expected_tasks,
            "actual_task_count": len(train_tasks),
            "tasks": train_tasks,
        },
    }


def write_blocker(final_dir: Path, prep: Dict[str, Any], run_command: str) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    blocker_path = final_dir / "BLOCKED_REBUILD_DO_NOT_USE.md"
    lines = [
        "# ALFWorld final KB rebuild blocked",
        "",
        "This directory is not safe for final evaluation. The final train KB was not rebuilt.",
        "",
        "## Blockers",
        "",
    ]
    for item in prep["blockers"]:
        lines.append(f"- {item}")
    if prep["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for item in prep["warnings"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Rebuild command to run after blockers are fixed",
            "",
            "```bash",
            run_command,
            "```",
            "",
        ]
    )
    blocker_path.write_text("\n".join(lines), encoding="utf-8")


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    final_dir = Path(args.final_dir)
    manifest_path = Path(args.manifest)
    prep = check_environment(args)
    status = "ready_to_rebuild_not_run" if prep["ready"] else "blocked_rebuild_not_started"
    manifest = build_manifest(args, prep, status=status, safe_for_eval=False)

    final_dir.mkdir(parents=True, exist_ok=True)
    write_json(final_dir / "prepare_summary.json", manifest)
    write_json(manifest_path, manifest)

    run_cmd = (
        "python3 scripts/alfworld_final_kb_builder.py --mode run "
        f"--final-dir {rel(final_dir)} --manifest {rel(manifest_path)} "
        f"--expected-tasks {args.expected_tasks} --max-trials {args.max_trials} "
        f"--model {args.model} --embedding-provider {args.embedding_provider} --seed {args.seed}"
    )
    if args.alfworld_data:
        run_cmd += f" --alfworld-data {args.alfworld_data}"

    if not prep["ready"]:
        write_blocker(final_dir, prep, run_cmd)

    print(json.dumps(manifest["prepare"], indent=2))
    print(json.dumps(manifest["scope_estimate"], indent=2))
    return manifest


def prompt_key_for(task_id: str) -> str:
    first = task_id.split("/", 1)[0]
    for prefix, key in PROMPT_KEYS:
        if prefix in first:
            return key
    return "react_put_0"


def extract_task_desc(observation: str) -> str:
    if "Your task is to:" in observation:
        return observation.split("Your task is to:", 1)[1].strip().splitlines()[0].strip()
    for line in observation.splitlines():
        if line.strip():
            return line.strip()
    return ""


def load_prompts() -> Dict[str, str]:
    prompts_path = repo_root() / "data" / "alfworld" / "prompts" / "alfworld_3prompts.json"
    return read_json(prompts_path)


def load_base_config(data_root: str) -> Dict[str, Any]:
    import yaml

    config_path = repo_root() / "data" / "alfworld" / "base_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    os.environ["ALFWORLD_DATA"] = data_root
    config["general"]["random_seed"] = int(config["general"].get("random_seed", 42))
    config["general"]["use_cuda"] = False
    return config


def run_trial(
    *,
    task: Dict[str, Any],
    trial_num: int,
    memory: List[str],
    config: Dict[str, Any],
    prompt: str,
    model: str,
    task_index: int,
    to_print: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.core.history import EnvironmentHistory
    from src.envs.alfworld_env import AlfworldEnv
    from src.frameworks.memory_retrieval_v2.agents.memory_allocation import MemoryAllocationReflexion
    from src.frameworks.react import action_stop_sequences, clean_action_text

    trial_config = json.loads(json.dumps(config))
    trial_config["dataset"]["eval_ood_data_path"] = task["task_path"]
    trial_config["general"]["evaluate"]["batch_size"] = 1

    env = AlfworldEnv(trial_config, split="eval_out_of_distribution")
    steps: List[Dict[str, Any]] = []
    usage_totals = Counter()
    reward = 0.0
    done = False
    try:
        ob, _info = env.reset()
        task_desc = extract_task_desc(ob)
        stabilized_prompt = (
            "You are an AI agent playing a text-based game. Follow the exact format "
            "of the examples below to solve the task.\n\n" + prompt
        )
        env_history = EnvironmentHistory(stabilized_prompt, ob, memory[-3:])
        framework = MemoryAllocationReflexion(model=model, to_print=to_print, success_threshold=SUCCESS_THRESHOLD)
        stop = action_stop_sequences(False)

        for cur_step in range(49):
            action_text, usage = framework._llm(str(env_history) + "Action:", stop=stop)
            usage_totals.update(usage or {})
            action = clean_action_text(action_text)
            env_history.add("action", action)
            observation, reward, done, info = env.step(action)
            if action.startswith("think:"):
                observation = "OK."
            env_history.add("observation", observation)
            steps.append(
                {
                    "step": cur_step + 1,
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
            if done or env_history.check_is_exhausted():
                break

        success = reward >= SUCCESS_THRESHOLD
        reflexion = ""
        reflexion_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
        if not success:
            before = time.time()
            reflexion = framework._generate_reflexion(str(env_history), memory)
            reflexion_usage["elapsed_seconds"] = round(time.time() - before, 3)

        trajectory = {
            "task_id": task["task_id"],
            "task_index": task_index,
            "task_desc": task_desc,
            "split": "train",
            "framework": "react_reflexion",
            "model": model,
            "trial_num": trial_num,
            "success": success,
            "is_success": success,
            "reward": reward,
            "game_file": task["game_file"],
            "task_path": task["task_path"],
            "steps": steps,
            "usage": dict(usage_totals),
        }
        reflexion_entry = {
            "task_id": task["task_id"],
            "task_index": task_index,
            "split": "train",
            "framework": "react_reflexion",
            "model": model,
            "trial_num": trial_num,
            "is_success": success,
            "success": success,
            "reward": reward,
            "reflexion": reflexion,
            "usage": reflexion_usage,
        }
        return trajectory, reflexion_entry
    finally:
        env.close()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_rebuild(args: argparse.Namespace) -> None:
    manifest = prepare(args)
    if not manifest["prepare"]["ready"]:
        raise SystemExit("Prepare checks failed; expensive rebuild was not started.")

    final_dir = Path(args.final_dir)
    run_dir = final_dir / "raw_react_reflexion_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LTM_EMBEDDING_PROVIDER"] = args.embedding_provider
    data_root = manifest["prepare"]["data_root"]
    config = load_base_config(data_root)
    prompts = load_prompts()
    trajectories: List[Dict[str, Any]] = []
    reflexions: List[Dict[str, Any]] = []
    task_memories: Dict[str, List[str]] = defaultdict(list)
    task_success: Dict[str, bool] = defaultdict(bool)

    suite_config = {
        "artifact_id": ARTIFACT_ID,
        "git_commit": git_commit(),
        "command": command_line(),
        "created_at_utc": utc_now(),
        "split": "train",
        "frameworks": ["react_reflexion"],
        "framework": "react_reflexion",
        "model": args.model,
        "embedding_provider": args.embedding_provider,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "seed": args.seed,
        "max_trials": args.max_trials,
        "expected_tasks": args.expected_tasks,
        "train_manifest_path": rel(Path(args.manifest)),
    }
    write_json(run_dir / "suite_config.json", suite_config)

    tasks = manifest["train_manifest"]["tasks"]
    for trial_num in range(1, args.max_trials + 1):
        for task_index, task in enumerate(tasks):
            if task_success[task["task_id"]]:
                continue
            prompt = prompts.get(prompt_key_for(task["task_id"]), prompts.get("react_put_0", ""))
            trajectory, reflexion_entry = run_trial(
                task=task,
                trial_num=trial_num,
                memory=task_memories[task["task_id"]],
                config=config,
                prompt=prompt,
                model=args.model,
                task_index=task_index,
                to_print=args.verbose,
            )
            trajectories.append(trajectory)
            reflexions.append(reflexion_entry)
            task_success[task["task_id"]] = bool(trajectory["success"])
            if reflexion_entry.get("reflexion"):
                task_memories[task["task_id"]].append(reflexion_entry["reflexion"])
            write_json(run_dir / "trajectories.json", trajectories)
            write_json(run_dir / "reflexions.json", reflexions)

    metrics = {
        "total_tasks": len(tasks),
        "completed_tasks": sum(1 for t in tasks if task_success[t["task_id"]]),
        "trajectory_rows": len(trajectories),
        "reflexion_rows": len(reflexions),
        "success_rate": (sum(1 for t in tasks if task_success[t["task_id"]]) / len(tasks)) if tasks else 0.0,
    }
    write_json(run_dir / "metrics.json", metrics)

    from src.frameworks.memory_retrieval_v2.preprocessing.knowledge_base_v2.script import generate_knowledge_base
    from src.frameworks.memory_retrieval_v2.retrieval.core.embedding_cache import create_knowledge_base_embeddings
    from src.frameworks.memory_retrieval_v2.retrieval.core.learning_counts import build_learning_counts_table

    generate_knowledge_base(str(run_dir), env="alfworld", allow_eval_trajectories=False)
    run_kb = run_dir / "knowledge_base.json"
    if not run_kb.exists():
        raise SystemExit("KB extraction did not produce knowledge_base.json")

    create_knowledge_base_embeddings(str(run_kb), force=True, embed_field="issue_text")
    build_learning_counts_table(str(run_kb), force=True)

    copy_if_exists(run_kb, final_dir / "knowledge_base.json")
    copy_if_exists(run_dir / "knowledge_base.csv", final_dir / "knowledge_base.csv")
    copy_if_exists(run_dir / "knowledge_base_progress.json", final_dir / "knowledge_base_progress.json")
    copy_if_exists(run_dir / "knowledge_base.issue_embeddings_cache.json", final_dir / "knowledge_base.issue_embeddings_cache.json")
    copy_if_exists(run_dir / "knowledge_base.mem_learning_counts.json", final_dir / "knowledge_base.mem_learning_counts.json")
    audit(args)


def iter_ref_trials(entry: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    for key in ("issue_ref", "evidence_ref"):
        ref = entry.get(key) or {}
        task_id = ref.get("task_id", "")
        try:
            trial_num = int(ref.get("trial_num", 0))
        except Exception:
            trial_num = 0
        yield task_id, trial_num


def sha256_or_md5_source_hash(path: Path, expected: str) -> str:
    """Return a source hash with the algorithm used by an existing cache."""
    if len(expected) == 32:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return sha256_file(path)


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    final_dir = Path(args.final_dir)
    manifest_path = Path(args.manifest)
    kb_path = final_dir / "knowledge_base.json"
    run_dir = final_dir / "raw_react_reflexion_run"
    trajectories_path = run_dir / "trajectories.json"
    reflexions_path = run_dir / "reflexions.json"
    errors: List[str] = []
    warnings: List[str] = []

    for path in (manifest_path, kb_path, trajectories_path, reflexions_path):
        if not path.exists():
            errors.append(f"Missing required file: {rel(path)}")

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    kb = read_json(kb_path) if kb_path.exists() else []
    trajectories = read_json(trajectories_path) if trajectories_path.exists() else []
    reflexions = read_json(reflexions_path) if reflexions_path.exists() else []
    train_tasks = manifest.get("train_manifest", {}).get("tasks", [])
    train_ids = {t.get("task_id") for t in train_tasks}

    traj_ids = {t.get("task_id") for t in trajectories}
    traj_trial_nums = [int(t.get("trial_num", 0) or 0) for t in trajectories]
    traj_split_values = Counter(str(t.get("split", "")) for t in trajectories)
    if len(train_ids) != args.expected_tasks:
        errors.append(f"Manifest has {len(train_ids)} train task IDs, expected {args.expected_tasks}.")
    if traj_ids != train_ids:
        errors.append(
            f"Trajectory task IDs do not match manifest: missing={len(train_ids - traj_ids)}, extra={len(traj_ids - train_ids)}."
        )
    if traj_trial_nums and max(traj_trial_nums) > args.max_trials:
        errors.append(f"Trajectory max trial is {max(traj_trial_nums)}, expected <= {args.max_trials}.")
    if set(traj_split_values) != {"train"}:
        errors.append(f"Trajectory split values are {dict(traj_split_values)}, expected only train.")

    success_mismatch = []
    reward_out_of_range = []
    for t in trajectories:
        reward = float(t.get("reward", 0.0) or 0.0)
        success = bool(t.get("success", t.get("is_success", False)))
        if success != (reward >= SUCCESS_THRESHOLD):
            success_mismatch.append({"task_id": t.get("task_id"), "trial_num": t.get("trial_num"), "reward": reward, "success": success})
        if reward < 0.0 or reward > 1.0:
            reward_out_of_range.append({"task_id": t.get("task_id"), "trial_num": t.get("trial_num"), "reward": reward})
    if success_mismatch:
        errors.append(f"{len(success_mismatch)} trajectories have success flag != reward >= {SUCCESS_THRESHOLD}.")
    if reward_out_of_range:
        errors.append(f"{len(reward_out_of_range)} trajectories have reward outside [0, 1].")

    valid_level_counts = Counter(str(e.get("valid_level", "")) for e in kb)
    malformed = {
        "invalid_valid_level": [],
        "empty_learning_text": [],
        "empty_issue_text": [],
        "missing_ref_task_id": [],
        "ref_task_not_train": [],
        "ref_trial_gt_max": [],
    }
    seen_unique_ids = set()
    duplicate_unique_ids = []
    for idx, entry in enumerate(kb):
        uid = str(entry.get("unique_id", idx + 1))
        if uid in seen_unique_ids:
            duplicate_unique_ids.append(uid)
        seen_unique_ids.add(uid)
        if entry.get("valid_level") not in VALID_LEVELS:
            malformed["invalid_valid_level"].append(uid)
        if not str(entry.get("learning_text", "")).strip():
            malformed["empty_learning_text"].append(uid)
        if not str(entry.get("issue_text", "")).strip():
            malformed["empty_issue_text"].append(uid)
        for ref_task_id, ref_trial in iter_ref_trials(entry):
            if not ref_task_id:
                malformed["missing_ref_task_id"].append(uid)
            elif ref_task_id not in train_ids:
                malformed["ref_task_not_train"].append(uid)
            if ref_trial > args.max_trials:
                malformed["ref_trial_gt_max"].append(uid)
    if duplicate_unique_ids:
        errors.append(f"Duplicate unique_id values: {duplicate_unique_ids[:10]}")
    for key, values in malformed.items():
        if values:
            errors.append(f"Malformed KB rows for {key}: {len(values)}.")

    split_counts = manifest.get("prepare", {}).get("split_counts", {})
    overlap_counts = manifest.get("prepare", {}).get("overlap_counts", {})
    if overlap_counts.get("train_valid_seen", 0) or overlap_counts.get("train_valid_unseen", 0):
        errors.append(f"Train/eval overlap counts are nonzero: {overlap_counts}.")
    if split_counts.get("train") != args.expected_tasks:
        errors.append(f"Prepared train split count is {split_counts.get('train')}, expected {args.expected_tasks}.")

    cache_checks = {}
    for name in ("knowledge_base.issue_embeddings_cache.json", "knowledge_base.mem_learning_counts.json"):
        path = final_dir / name
        if not path.exists():
            errors.append(f"Missing cache file: {rel(path)}")
            continue
        try:
            cache = read_json(path)
            cache_checks[name] = {
                "source_hash_matches": cache.get("source_hash") == sha256_or_md5_source_hash(kb_path, cache.get("source_hash", "")),
                "embedding_model": cache.get("embedding_model"),
                "entry_count": cache.get("entry_count", cache.get("total_learning_count")),
            }
        except Exception as exc:
            errors.append(f"Corrupt cache file {rel(path)}: {exc}")

    audit_summary = {
        "artifact_id": ARTIFACT_ID,
        "audited_at_utc": utc_now(),
        "git_commit": git_commit(),
        "safe_for_eval": not errors,
        "errors": errors,
        "warnings": warnings,
        "kb_path": rel(kb_path),
        "run_dir": rel(run_dir),
        "manifest_path": rel(manifest_path),
        "counts": {
            "manifest_train_tasks": len(train_ids),
            "trajectory_rows": len(trajectories),
            "trajectory_unique_task_ids": len(traj_ids),
            "reflexion_rows": len(reflexions),
            "kb_entries": len(kb),
            "valid_level_counts": dict(valid_level_counts),
            "trajectory_split_values": dict(traj_split_values),
            "trajectory_max_trial": max(traj_trial_nums) if traj_trial_nums else None,
        },
        "malformed": {k: v[:50] for k, v in malformed.items()},
        "success_mismatch_examples": success_mismatch[:20],
        "reward_out_of_range_examples": reward_out_of_range[:20],
        "cache_checks": cache_checks,
    }
    write_json(final_dir / "audit_summary.json", audit_summary)

    if manifest_path.exists():
        manifest["status"] = "safe_for_eval" if audit_summary["safe_for_eval"] else "audit_failed"
        manifest["safe_for_eval"] = audit_summary["safe_for_eval"]
        manifest["audit_summary_path"] = rel(final_dir / "audit_summary.json")
        write_json(manifest_path, manifest)

    print(json.dumps(audit_summary, indent=2))
    if errors:
        raise SystemExit("ALFWorld final KB audit failed.")
    return audit_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "run", "audit"], default="prepare")
    parser.add_argument("--final-dir", default=str(DEFAULT_FINAL_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--alfworld-data", default="")
    parser.add_argument("--expected-tasks", type=int, default=DEFAULT_EXPECTED_TASKS)
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        manifest = prepare(args)
        if not manifest["prepare"]["ready"]:
            raise SystemExit(2)
        return
    if args.mode == "run":
        run_rebuild(args)
        return
    if args.mode == "audit":
        audit(args)
        return
    raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
