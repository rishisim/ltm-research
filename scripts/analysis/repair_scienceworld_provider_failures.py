#!/usr/bin/env python3
"""Archive and clear suspected provider-failure rows for targeted ScienceWorld reruns."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set


FILES_TO_FILTER = (
    "attempts.json",
    "trajectories.json",
    "agent_trajectories.jsonl",
    "trajectories.jsonl",
    "knowledge_retrieval_bases.json",
    "knowledge_retrieval_bases.jsonl",
)


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def task_ids_for_record(record: Any) -> Set[str]:
    ids: Set[str] = set()
    if not isinstance(record, dict):
        return ids

    for key in ("task_id", "task_id_str"):
        value = record.get(key)
        if isinstance(value, str) and value:
            ids.add(value)

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("task_id")
        if isinstance(value, str) and value:
            ids.add(value)

    for key, value in record.items():
        if isinstance(key, str) and key.startswith("scienceworld_") and isinstance(value, dict):
            ids.add(key)
    return ids


def filter_json(path: Path, bad_task_ids: Set[str]) -> Dict[str, Any]:
    payload = load_json(path)
    before = len(payload) if hasattr(payload, "__len__") else None
    removed = 0

    if isinstance(payload, list):
        kept = []
        for row in payload:
            if task_ids_for_record(row) & bad_task_ids:
                removed += 1
                continue
            kept.append(row)
        payload = kept
    elif isinstance(payload, dict):
        for task_id in list(payload.keys()):
            if task_id in bad_task_ids:
                removed += 1
                del payload[task_id]
    else:
        return {"path": str(path), "before": before, "after": before, "removed": 0}

    write_json(path, payload)
    after = len(payload) if hasattr(payload, "__len__") else None
    return {"path": str(path), "before": before, "after": after, "removed": removed}


def filter_jsonl(path: Path, bad_task_ids: Set[str]) -> Dict[str, Any]:
    kept_lines: List[str] = []
    before = 0
    removed = 0
    with path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            before += 1
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            if task_ids_for_record(payload) & bad_task_ids:
                removed += 1
                continue
            kept_lines.append(json.dumps(payload, separators=(",", ":")) + "\n")

    with path.open("w") as f:
        f.writelines(kept_lines)
    return {"path": str(path), "before": before, "after": len(kept_lines), "removed": removed}


def unique_archive_dir(archive_root: Path, framework: str, seed: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = archive_root / timestamp / framework / seed
    index = 1
    while candidate.exists():
        candidate = archive_root / f"{timestamp}_{index}" / framework / seed
        index += 1
    return candidate


def runs_from_audit(audit: Dict[str, Any], env: str, seed: str) -> List[Dict[str, Any]]:
    env_report = audit.get("environments", {}).get(env, {})
    runs = env_report.get("runs", {}) if isinstance(env_report, dict) else {}
    selected: List[Dict[str, Any]] = []
    for run_key, summary in runs.items():
        if not isinstance(summary, dict):
            continue
        if not run_key.endswith(f"/seed_{seed}"):
            continue
        bad_task_ids = summary.get("suspected_llm_failure_task_ids") or []
        if not bad_task_ids:
            continue
        framework = run_key.split("/seed_")[0]
        selected.append(
            {
                "framework": framework,
                "seed": f"seed_{seed}",
                "run_dir": Path(str(summary["run_dir"])),
                "bad_task_ids": sorted(set(str(item) for item in bad_task_ids)),
            }
        )
    return selected


def repair_run(run: Dict[str, Any], archive_root: Path, apply: bool) -> Dict[str, Any]:
    run_dir = Path(run["run_dir"])
    framework = str(run["framework"])
    seed = str(run["seed"])
    bad_task_ids = set(run["bad_task_ids"])
    archive_dir = unique_archive_dir(archive_root, framework, seed)
    record: Dict[str, Any] = {
        **run,
        "run_dir": str(run_dir),
        "archive_dir": str(archive_dir),
        "files": [],
        "applied": apply,
    }
    if not apply:
        return record

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, archive_dir)

    for name in FILES_TO_FILTER:
        path = run_dir / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            file_record = filter_jsonl(path, bad_task_ids)
        else:
            file_record = filter_json(path, bad_task_ids)
        record["files"].append(file_record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive ScienceWorld provider-failure outputs and clear only affected rows."
    )
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--env", default="scienceworld")
    parser.add_argument("--seed", default="2")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    audit = load_json(args.audit_json)
    runs = runs_from_audit(audit, args.env, args.seed)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_json": str(args.audit_json),
        "archive_root": str(args.archive_root),
        "env": args.env,
        "seed": args.seed,
        "run_count": len(runs),
        "runs": [repair_run(run, args.archive_root, args.apply) for run in runs],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.manifest, manifest)
    print(json.dumps({"apply": args.apply, "run_count": len(runs), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
