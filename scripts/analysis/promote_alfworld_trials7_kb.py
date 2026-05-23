#!/usr/bin/env python3
"""Audit and promote the legacy ALFWorld KB as a final trials7 artifact.

This script is intentionally offline: it never runs ALFWorld tasks or calls an
LLM. It verifies provenance from existing artifacts, removes malformed rows
from the promoted KB, and writes audit/provenance files for the final campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_TYPES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)

VALID_LEVELS = ("VALID_NEXT_TRIAL", "VALID_SAME_TRIAL", "CANDIDATE")
NON_INFORMATIVE_LEARNINGS = {
    "",
    "n/a",
    "none",
    "no specific learning extracted.",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def normalize_task_id(raw: Any) -> str:
    text = str(raw or "").strip()
    if ":" in text and text.split(":", 1)[0] in {"train", "valid_seen", "valid_unseen"}:
        text = text.split(":", 1)[1]
    return text


def base_task_id(task_id: str) -> str:
    return normalize_task_id(task_id).split("/", 1)[0]


def task_type_for(task_id: str) -> str:
    base = base_task_id(task_id)
    for task_type in TASK_TYPES:
        if base.startswith(task_type):
            return task_type
    return "unknown"


def parse_trial_num(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        match = re.search(r"\btrial[_-]?(\d+)\b", stripped, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def load_official_split_index(cache_root: Path) -> dict[str, Any]:
    full_to_split: dict[str, str] = {}
    base_to_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = {}
    present = cache_root.exists()

    for split in ("train", "valid_seen", "valid_unseen"):
        split_root = cache_root / split
        counts[split] = 0
        if not split_root.exists():
            present = False
            continue
        for game_file in split_root.rglob("game.tw-pddl"):
            rel = game_file.parent.relative_to(split_root).as_posix()
            full_to_split[rel] = split
            base_to_splits[base_task_id(rel)].add(split)
            counts[split] += 1

    return {
        "cache_root": str(cache_root),
        "available": present,
        "counts": counts,
        "full_to_split": full_to_split,
        "base_to_splits": {k: sorted(v) for k, v in base_to_splits.items()},
    }


def classify_split(task_id: str, split_index: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_task_id(task_id)
    if not normalized:
        return {"split": "missing", "match_type": "missing", "matches": []}

    full_to_split = split_index["full_to_split"]
    base_to_splits = split_index["base_to_splits"]

    if normalized in full_to_split:
        split = full_to_split[normalized]
        return {"split": split, "match_type": "full_task_id", "matches": [split]}

    base = base_task_id(normalized)
    matches = base_to_splits.get(base, [])
    if len(matches) == 1:
        return {"split": matches[0], "match_type": "base_task_id", "matches": matches}
    if len(matches) > 1:
        return {"split": "ambiguous", "match_type": "base_task_id", "matches": matches}
    return {"split": "unknown", "match_type": "none", "matches": []}


def ref_task_id(row: dict[str, Any], ref_name: str) -> str:
    ref = row.get(ref_name)
    if isinstance(ref, dict):
        return normalize_task_id(ref.get("task_id", ""))
    return ""


def ref_trial(row: dict[str, Any], ref_name: str) -> int | None:
    ref = row.get(ref_name)
    if isinstance(ref, dict):
        return parse_trial_num(ref.get("trial_num"))
    return None


def quarantine_reasons(
    row: Any,
    seen_unique_ids: set[str],
    split_index: dict[str, Any],
    max_trials: int,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(row, dict):
        return ["row_not_object"]

    unique_id = str(row.get("unique_id", "")).strip()
    if not unique_id:
        reasons.append("missing_unique_id")
    elif unique_id in seen_unique_ids:
        reasons.append("duplicate_unique_id")

    task_desc = str(row.get("task_desc", "") or "").strip()
    learning = str(row.get("learning_text", "") or "").strip()
    valid_level = str(row.get("valid_level", "") or "").strip()

    if not task_desc:
        reasons.append("missing_task_desc")
    if learning.lower() in NON_INFORMATIVE_LEARNINGS:
        reasons.append("non_informative_learning_text")
    if valid_level not in VALID_LEVELS:
        reasons.append("invalid_valid_level")

    for ref_name in ("issue_ref", "evidence_ref"):
        ref = row.get(ref_name)
        if not isinstance(ref, dict):
            reasons.append(f"missing_{ref_name}")
            continue

        task_id = normalize_task_id(ref.get("task_id", ""))
        if not task_id:
            reasons.append(f"missing_{ref_name}_task_id")
        else:
            split_result = classify_split(task_id, split_index)
            if split_result["split"] != "train":
                reasons.append(f"{ref_name}_split_{split_result['split']}")

        trial_num = parse_trial_num(ref.get("trial_num"))
        if trial_num is None:
            reasons.append(f"missing_{ref_name}_trial_num")
        elif trial_num < 1 or trial_num > max_trials:
            reasons.append(f"{ref_name}_trial_out_of_range")

    return reasons


def ref_max_trial(row: dict[str, Any]) -> int | None:
    trials = [trial for trial in (ref_trial(row, "issue_ref"), ref_trial(row, "evidence_ref")) if trial is not None]
    return max(trials) if trials else None


def source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for label, path in paths.items():
        if path.exists():
            hashes[label] = file_sha256(path)
    return hashes


def build_filtered_issue_cache(
    source_cache_path: Path,
    final_kb_path: Path,
    clean_source_indices: list[int],
    final_rows: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any] | None:
    if not source_cache_path.exists():
        return None
    source_cache = read_json(source_cache_path)
    source_entries = source_cache.get("entries", [])
    if len(source_entries) <= max(clean_source_indices, default=-1):
        return None

    entries = []
    for new_index, source_index in enumerate(clean_source_indices):
        source_entry = deepcopy(source_entries[source_index])
        source_entry["index"] = new_index
        source_entry["valid_level"] = final_rows[new_index].get("valid_level", "CANDIDATE")
        source_entry["goal_phase"] = final_rows[new_index].get("goal_phase", "")
        entries.append(source_entry)

    return {
        "source_hash": file_md5(final_kb_path),
        "source_path": str(final_kb_path),
        "created_at": generated_at,
        "embedding_model": source_cache.get("embedding_model", "gemini-embedding-001"),
        "embed_field": source_cache.get("embed_field", "issue_text"),
        "entry_count": len(entries),
        "entries": entries,
        "derived_from": str(source_cache_path),
    }


def build_filtered_learning_counts(
    source_counts_path: Path,
    final_kb_path: Path,
    final_rows: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any] | None:
    if not source_counts_path.exists():
        return None
    source_counts = read_json(source_counts_path)
    embedding_by_task = {
        entry.get("task_desc", ""): entry.get("task_desc_embedding", [])
        for entry in source_counts.get("entries", [])
    }

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "learning_count": 0,
            "entry_indices": [],
            "valid_levels": [],
            "goal_phases": [],
        }
    )
    for index, row in enumerate(final_rows):
        task_desc = str(row.get("task_desc", "") or "").strip()
        if not task_desc:
            continue
        groups[task_desc]["learning_count"] += 1
        groups[task_desc]["entry_indices"].append(index)
        groups[task_desc]["valid_levels"].append(row.get("valid_level", "CANDIDATE"))
        groups[task_desc]["goal_phases"].append(row.get("goal_phase", ""))

    entries = []
    missing_embeddings = []
    for task_desc, group in groups.items():
        embedding = embedding_by_task.get(task_desc)
        if not embedding:
            missing_embeddings.append(task_desc)
            continue
        validated_count = sum(1 for level in group["valid_levels"] if level in {"VALID_SAME_TRIAL", "VALID_NEXT_TRIAL"})
        entries.append(
            {
                "task_desc": task_desc,
                "task_desc_embedding": embedding,
                "learning_count": group["learning_count"],
                "validated_learning_count": validated_count,
                "candidate_count": group["learning_count"] - validated_count,
                "entry_indices": group["entry_indices"],
                "goal_phases": sorted({phase for phase in group["goal_phases"] if phase}),
            }
        )

    if missing_embeddings:
        return None

    entries.sort(key=lambda item: item["learning_count"], reverse=True)
    return {
        "source_hash": file_md5(final_kb_path),
        "source_path": str(final_kb_path),
        "created_at": generated_at,
        "embedding_model": source_counts.get("embedding_model", "gemini-embedding-001"),
        "unique_task_count": len(entries),
        "total_learning_count": len(final_rows),
        "entries": entries,
        "derived_from": str(source_counts_path),
    }


def load_manifest_tasks(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = read_json(path)
    tasks = data.get("tasks", []) if isinstance(data, dict) else data
    return {normalize_task_id(item) for item in tasks}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def render_report(manifest: dict[str, Any], audit: dict[str, Any]) -> str:
    source_counts = audit["source"]["valid_level_counts"]
    promoted_counts = audit["promoted"]["valid_level_counts"]
    quarantine_counts = audit["quarantine"]["reason_counts"]
    coverage = audit["coverage"]
    split = audit["split_audit"]

    report = [
        "# ALFWorld Trials7 KB Audit and Promotion Report",
        "",
        f"Status: {'promoted, safe_for_eval true' if manifest['safe_for_eval'] else 'blocked, not safe for eval'}.",
        "",
        f"Worktree: `{manifest['worktree']}`",
        "",
        f"Branch: `{manifest['branch']}`",
        "",
        f"Generation git commit: `{manifest['generation_git_commit']}`",
        "",
        "## Final Artifact",
        "",
        f"- Final KB path: `{manifest['final']['kb_path']}`",
        f"- Manifest path: `{manifest['final']['manifest_path']}`",
        f"- Quarantine path: `{manifest['final']['quarantine_path']}`",
        f"- Audit summary path: `{manifest['final']['audit_summary_path']}`",
        "",
        "## Protocol",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["offline framework", manifest["protocol"]["framework"]],
                ["split", manifest["protocol"]["split"]],
                ["max_trials", manifest["protocol"]["max_trials"]],
                ["model", manifest["protocol"]["model"]],
                ["embedding_provider", manifest["protocol"]["embedding_provider"]],
                ["embedding_model", manifest["protocol"]["embedding_model"]],
                ["eval max_learnings", manifest["protocol"]["max_learnings"]],
                ["eval min_valid_level", manifest["protocol"]["min_valid_level"]],
                ["seed", manifest["protocol"]["seed"]],
            ],
        ),
        "",
        "## Train-Only Provenance",
        "",
        f"- Method: `{split['method']}`",
        f"- Official split root: `{split['official_split_root']}`",
        f"- Official split counts: train={split['official_counts'].get('train')}, valid_seen={split['official_counts'].get('valid_seen')}, valid_unseen={split['official_counts'].get('valid_unseen')}",
        f"- Trajectory full task IDs mapped to train: {split['trajectory_train_count']}/{split['trajectory_unique_task_ids']}",
        f"- Trajectory overlap with valid_seen: {split['trajectory_valid_seen_overlap']}",
        f"- Trajectory overlap with valid_unseen: {split['trajectory_valid_unseen_overlap']}",
        f"- Promoted KB refs with non-train or unknown split: {split['promoted_non_train_ref_count']}",
        f"- Limitation: {split['limitation']}",
        "",
        "## Coverage",
        "",
        f"- Intended train games: {coverage['intended_train_games']}",
        f"- Observed unique trajectory task IDs: {coverage['observed_unique_trajectory_task_ids']}",
        f"- Coverage complete: {coverage['coverage_complete']}",
        "",
        markdown_table(
            ["Task type", "Observed unique task IDs", "Intended"],
            [
                [task_type, coverage["observed_by_task_type"].get(task_type, 0), 20]
                for task_type in TASK_TYPES
            ],
        ),
        "",
        "## Row Audit",
        "",
        f"- Source KB entries: {audit['source']['kb_entries']}",
        f"- Promoted clean entries: {audit['promoted']['kb_entries']}",
        f"- Quarantined entries: {audit['quarantine']['row_count']}",
        f"- Max observed source trial: {audit['source']['max_ref_trial']}",
        f"- Max promoted reference trial: {audit['promoted']['max_ref_trial']}",
        "",
        "Source valid levels:",
        "",
        markdown_table(["valid_level", "count"], [[k or "blank", v] for k, v in source_counts.items()]),
        "",
        "Promoted valid levels:",
        "",
        markdown_table(["valid_level", "count"], [[k, v] for k, v in promoted_counts.items()]),
        "",
        "Quarantine reasons:",
        "",
        markdown_table(["reason", "count"], [[k, v] for k, v in quarantine_counts.items()]),
        "",
        "## Commands",
        "",
    ]

    for command in manifest["commands"]:
        report.append(f"- `{command}`")

    report.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in manifest["caveats"]:
        report.append(f"- {caveat}")
    if not manifest["caveats"]:
        report.append("- None.")
    report.append("")
    return "\n".join(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-kb", default="alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json")
    parser.add_argument("--source-trajectories", default="alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/trajectories.json")
    parser.add_argument("--source-reflexions", default="alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/reflexions.json")
    parser.add_argument("--source-progress", default="alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/knowledge_base_progress.json")
    parser.add_argument("--source-issue-cache", default="alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.issue_embeddings_cache.json")
    parser.add_argument("--source-learning-counts", default="alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.mem_learning_counts.json")
    parser.add_argument("--official-alfworld-root", default="/Users/rishisim/.cache/alfworld/json_2.1.1")
    parser.add_argument("--output-dir", default="final_runs/kb/alfworld_train_reflexion_trials7")
    parser.add_argument("--manifest", default="final_runs/manifests/alfworld_train_reflexion_trials7.json")
    parser.add_argument("--report", default="experiment_reports/final/alfworld_kb_trials7_report.md")
    parser.add_argument("--max-trials", type=int, default=7)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--embedding-provider", default="gemini")
    parser.add_argument("--embedding-model", default="gemini-embedding-001")
    args = parser.parse_args()

    repo = Path.cwd()
    generated_at = utc_now()
    source_kb_path = Path(args.source_kb)
    source_trajectories_path = Path(args.source_trajectories)
    source_reflexions_path = Path(args.source_reflexions)
    source_progress_path = Path(args.source_progress)
    source_issue_cache_path = Path(args.source_issue_cache)
    source_learning_counts_path = Path(args.source_learning_counts)
    output_dir = Path(args.output_dir)
    final_kb_path = output_dir / "knowledge_base.json"
    quarantine_path = output_dir / "knowledge_base.quarantine.json"
    audit_summary_path = output_dir / "audit_summary.json"
    provenance_path = output_dir / "provenance.json"
    manifest_path = Path(args.manifest)
    report_path = Path(args.report)

    source_kb = read_json(source_kb_path)
    trajectories = read_json(source_trajectories_path)
    reflexions = read_json(source_reflexions_path)
    progress = read_json(source_progress_path)

    if not isinstance(source_kb, list):
        raise TypeError(f"Expected source KB list, got {type(source_kb).__name__}")
    if not isinstance(trajectories, list):
        raise TypeError(f"Expected trajectories list, got {type(trajectories).__name__}")

    split_index = load_official_split_index(Path(args.official_alfworld_root))
    official_available = bool(split_index["available"])

    clean_rows: list[dict[str, Any]] = []
    clean_source_indices: list[int] = []
    quarantined_rows: list[dict[str, Any]] = []
    seen_unique_ids: set[str] = set()
    source_ref_split_counts = Counter()
    promoted_ref_split_counts = Counter()

    for index, row in enumerate(source_kb):
        reasons = quarantine_reasons(row, seen_unique_ids, split_index, args.max_trials)
        if isinstance(row, dict):
            unique_id = str(row.get("unique_id", "")).strip()
            if unique_id:
                seen_unique_ids.add(unique_id)
            for ref_name in ("issue_ref", "evidence_ref"):
                task_id = ref_task_id(row, ref_name)
                source_ref_split_counts[classify_split(task_id, split_index)["split"]] += 1
        if reasons:
            quarantined_rows.append({"source_index": index, "reasons": reasons, "row": row})
        else:
            promoted_row = deepcopy(row)
            clean_source_indices.append(index)
            clean_rows.append(promoted_row)
            for ref_name in ("issue_ref", "evidence_ref"):
                task_id = ref_task_id(promoted_row, ref_name)
                promoted_ref_split_counts[classify_split(task_id, split_index)["split"]] += 1

    write_json(final_kb_path, clean_rows)
    write_json(quarantine_path, quarantined_rows)

    issue_cache = build_filtered_issue_cache(
        source_issue_cache_path,
        final_kb_path,
        clean_source_indices,
        clean_rows,
        generated_at,
    )
    if issue_cache is not None:
        write_json(output_dir / "knowledge_base.issue_embeddings_cache.json", issue_cache)

    learning_counts = build_filtered_learning_counts(
        source_learning_counts_path,
        final_kb_path,
        clean_rows,
        generated_at,
    )
    if learning_counts is not None:
        write_json(output_dir / "knowledge_base.mem_learning_counts.json", learning_counts)

    source_valid_level_counts = Counter(str(row.get("valid_level", "") if isinstance(row, dict) else "") for row in source_kb)
    promoted_valid_level_counts = Counter(str(row.get("valid_level", "")) for row in clean_rows)
    quarantine_reason_counts = Counter(reason for item in quarantined_rows for reason in item["reasons"])

    trajectory_task_ids = [normalize_task_id(row.get("task_id", "")) for row in trajectories if isinstance(row, dict)]
    trajectory_unique_ids = sorted({task_id for task_id in trajectory_task_ids if task_id})
    trajectory_split_counts = Counter(classify_split(task_id, split_index)["split"] for task_id in trajectory_unique_ids)
    trajectory_trial_counts = Counter(str(row.get("trial_num", "")) for row in trajectories if isinstance(row, dict))
    trajectory_success_counts = Counter(str(bool(row.get("is_success", False))).lower() for row in trajectories if isinstance(row, dict))
    observed_by_task_type = Counter(task_type_for(task_id) for task_id in trajectory_unique_ids)

    valid_seen_manifest = load_manifest_tasks(Path("alfworld_runs/memory_retrieval_v2/memory_agent_runs/manifests/valid_seen_140_tasks.json"))
    valid_unseen_manifest = load_manifest_tasks(Path("alfworld_runs/memory_retrieval_v2/memory_agent_runs/manifests/valid_unseen_134_tasks.json"))
    trajectory_set = set(trajectory_unique_ids)

    source_trials = [trial for row in source_kb if isinstance(row, dict) for trial in [ref_max_trial(row)] if trial is not None]
    promoted_trials = [trial for row in clean_rows for trial in [ref_max_trial(row)] if trial is not None]

    source_paths = {
        "knowledge_base.json": source_kb_path,
        "trajectories.json": source_trajectories_path,
        "reflexions.json": source_reflexions_path,
        "knowledge_base_progress.json": source_progress_path,
        "knowledge_base.issue_embeddings_cache.json": source_issue_cache_path,
        "knowledge_base.mem_learning_counts.json": source_learning_counts_path,
    }

    promoted_non_train_refs = sum(v for k, v in promoted_ref_split_counts.items() if k != "train")
    max_source_trial = max(source_trials) if source_trials else None
    max_promoted_trial = max(promoted_trials) if promoted_trials else None
    coverage_complete = len(trajectory_unique_ids) == 120

    caveats: list[str] = []
    if not coverage_complete:
        caveats.append(
            "The original fixed 120-game train manifest is not present in repo-local artifacts; "
            f"legacy trajectories cover {len(trajectory_unique_ids)}/120 intended train games."
        )
    if any(count != 20 for task_type, count in observed_by_task_type.items() if task_type in TASK_TYPES):
        caveats.append("Observed coverage is uneven across task types; see coverage table.")
    if not official_available:
        caveats.append("Official ALFWorld split cache was unavailable; split audit used repo-local artifacts only.")
    if issue_cache is None:
        caveats.append("Issue embedding cache was not promoted because the source cache could not be remapped cleanly.")
    if learning_counts is None:
        caveats.append("Learning-count cache was not promoted because source task-desc embeddings could not be remapped cleanly.")
    if manifest_path.exists():
        caveats.append("This run overwrote a prior trials7 manifest in this worktree.")

    safe_for_eval = (
        official_available
        and len(clean_rows) > 0
        and promoted_valid_level_counts.get("VALID_NEXT_TRIAL", 0) > 0
        and promoted_non_train_refs == 0
        and max_source_trial is not None
        and max_source_trial <= args.max_trials
        and max_promoted_trial is not None
        and max_promoted_trial <= args.max_trials
    )

    command = "python3 " + " ".join(shlex.quote(part) for part in sys.argv)
    branch = git_value(repo, "branch", "--show-current")
    commit = git_value(repo, "rev-parse", "HEAD")
    status = git_value(repo, "status", "--short")

    audit = {
        "generated_at_utc": generated_at,
        "source": {
            "kb_entries": len(source_kb),
            "valid_level_counts": dict(source_valid_level_counts),
            "max_ref_trial": max_source_trial,
            "ref_split_counts": dict(source_ref_split_counts),
            "sha256": source_hashes(source_paths),
        },
        "promoted": {
            "kb_entries": len(clean_rows),
            "valid_level_counts": dict(promoted_valid_level_counts),
            "max_ref_trial": max_promoted_trial,
            "ref_split_counts": dict(promoted_ref_split_counts),
            "sha256": {
                "knowledge_base.json": file_sha256(final_kb_path),
                "knowledge_base.quarantine.json": file_sha256(quarantine_path),
            },
            "cache_files": {
                "issue_embeddings_cache": str(output_dir / "knowledge_base.issue_embeddings_cache.json") if issue_cache else None,
                "mem_learning_counts": str(output_dir / "knowledge_base.mem_learning_counts.json") if learning_counts else None,
            },
        },
        "quarantine": {
            "row_count": len(quarantined_rows),
            "reason_counts": dict(quarantine_reason_counts),
            "path": str(quarantine_path),
        },
        "coverage": {
            "intended_train_games": 120,
            "observed_unique_trajectory_task_ids": len(trajectory_unique_ids),
            "coverage_complete": coverage_complete,
            "observed_by_task_type": {task_type: observed_by_task_type.get(task_type, 0) for task_type in TASK_TYPES},
            "trajectory_rows": len(trajectories),
            "trajectory_trial_counts": dict(trajectory_trial_counts),
            "trajectory_success_counts": dict(trajectory_success_counts),
            "reflexion_rows": len(reflexions) if isinstance(reflexions, list) else None,
            "progress_entries": len(progress) if isinstance(progress, list) else None,
        },
        "split_audit": {
            "method": "official ALFWorld cache walk with base-task fallback for legacy refs",
            "official_split_root": str(args.official_alfworld_root),
            "official_available": official_available,
            "official_counts": split_index["counts"],
            "trajectory_unique_task_ids": len(trajectory_unique_ids),
            "trajectory_train_count": trajectory_split_counts.get("train", 0),
            "trajectory_split_counts": dict(trajectory_split_counts),
            "trajectory_valid_seen_overlap": len(trajectory_set & valid_seen_manifest),
            "trajectory_valid_unseen_overlap": len(trajectory_set & valid_unseen_manifest),
            "promoted_non_train_ref_count": promoted_non_train_refs,
            "promoted_ref_split_counts": dict(promoted_ref_split_counts),
            "limitation": (
                "Original sampled 120-game train manifest is absent; exact missing game IDs cannot be reconstructed "
                "from repo-local artifacts. Split membership is verified against the official ALFWorld cache."
            ),
        },
    }

    manifest = {
        "artifact_id": "alfworld_train_reflexion_trials7",
        "status": "promoted" if safe_for_eval else "blocked_not_promoted",
        "safe_for_eval": safe_for_eval,
        "generated_at_utc": generated_at,
        "branch": branch,
        "worktree": str(repo),
        "generation_git_commit": commit,
        "generation_git_status_short": status,
        "commands": [command],
        "protocol": {
            "environment": "ALFWorld",
            "split": "train",
            "framework": "react_reflexion",
            "framework_list": ["react_reflexion"],
            "max_trials": args.max_trials,
            "model": args.model,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "seed": "not_recorded_in_legacy_source",
            "max_learnings": "N/A_KB_generation_eval_fixed_later",
            "min_valid_level": "N/A_KB_generation_eval_uses_VALID_NEXT_TRIAL_later",
        },
        "source": {
            "run_directory": str(source_trajectories_path.parent),
            "kb_path": str(source_kb_path),
            "trajectories_path": str(source_trajectories_path),
            "reflexions_path": str(source_reflexions_path),
            "progress_path": str(source_progress_path),
            "issue_embeddings_cache_path": str(source_issue_cache_path),
            "learning_counts_cache_path": str(source_learning_counts_path),
        },
        "final": {
            "run_directory": str(output_dir),
            "kb_path": str(final_kb_path),
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "audit_summary_path": str(audit_summary_path),
            "provenance_path": str(provenance_path),
            "quarantine_path": str(quarantine_path),
        },
        "audit_summary": audit,
        "caveats": caveats,
    }

    write_json(audit_summary_path, audit)
    write_json(provenance_path, manifest)
    write_json(manifest_path, manifest)
    write_text(report_path, render_report(manifest, audit))

    if safe_for_eval:
        write_text(
            output_dir / "PROMOTED_SAFE_FOR_EVAL.md",
            "# ALFWorld Trials7 KB Promoted\n\n"
            "This directory contains the cleaned, train-only ALFWorld trials7 KB promoted from legacy artifacts.\n"
            "Use `knowledge_base.json` as the final eval memory bank with `min_valid_level=VALID_NEXT_TRIAL`.\n",
        )
        blocked_marker = output_dir / "BLOCKED_DO_NOT_USE.md"
        if blocked_marker.exists():
            blocked_marker.unlink()
    else:
        write_text(
            output_dir / "BLOCKED_DO_NOT_USE.md",
            "# ALFWorld Trials7 KB Blocked\n\n"
            "No ALFWorld trials7 KB in this directory is safe for final eval use. See `audit_summary.json` "
            "and `experiment_reports/final/alfworld_kb_trials7_report.md` for blockers.\n",
        )

    print(f"safe_for_eval={safe_for_eval}")
    print(f"promoted_rows={len(clean_rows)}")
    print(f"quarantined_rows={len(quarantined_rows)}")
    print(f"final_kb={final_kb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
