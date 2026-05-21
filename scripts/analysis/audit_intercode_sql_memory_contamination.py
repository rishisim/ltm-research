#!/usr/bin/env python3
"""Audit InterCode SQL KB and retrieval logs for runner/protocol contamination."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.utils.sql_memory_sanitizer import classify_memory, split_memory_bank


DEFAULT_KB = (
    "intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json"
)


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_retrieval_records(path: Path) -> Dict[str, Any]:
    if path.suffix == ".jsonl":
        records: Dict[str, Any] = {}
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task_id = record.get("task_id") or record.get("query") or str(len(records))
                records[str(task_id)] = record
        return records
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def selected_count(record: Dict[str, Any]) -> int:
    metadata = record.get("metadata") or {}
    for key in ("actual_selected_count", "pick_learning_count"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
    return 0


def classify_retrieval_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    entry = {
        "issue_text": (row.get("issue_ref") or {}).get("issue_text", "")
        or row.get("issue_text", ""),
        "learning_text": row.get("learning_text", ""),
        "obj_type": row.get("obj_type", ""),
        "verbs": row.get("verbs", ""),
        "goal_phase": row.get("goal_phase", ""),
        "task_desc": row.get("task_desc", ""),
    }
    return classify_memory(entry)


def audit_kb(kb_path: Path) -> Dict[str, Any]:
    memory_bank = load_json(kb_path)
    if not isinstance(memory_bank, list):
        raise ValueError(f"Expected a JSON list in {kb_path}")

    kept, quarantined, reason_counts = split_memory_bank(memory_bank)
    return {
        "path": str(kb_path),
        "total_memories": len(memory_bank),
        "kept_memories": len(kept),
        "quarantined_memories": len(quarantined),
        "reason_counts": dict(sorted(reason_counts.items())),
        "quarantined_unique_ids": [
            item.get("unique_id", "") for item in quarantined
        ],
    }


def iter_retrieval_paths(paths: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            resolved.extend(sorted(path.rglob("knowledge_retrieval_bases.json")))
            resolved.extend(sorted(path.rglob("knowledge_retrieval_bases.jsonl")))
        elif path.exists():
            resolved.append(path)
    return resolved


def audit_retrieval_file(path: Path) -> Dict[str, Any]:
    data = load_retrieval_records(path)

    contaminated_tasks = set()
    selected_contaminated_tasks = set()
    contaminated_rows = 0
    selected_contaminated_rows = 0
    reason_counts: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for task_id, record in data.items():
        base = record.get("knowledge_retrieval_base") or []
        pick_count = selected_count(record)
        for row_index, row in enumerate(base):
            contaminated, reasons = classify_retrieval_row(row)
            if not contaminated:
                continue
            contaminated_rows += 1
            contaminated_tasks.add(task_id)
            reason_counts.update(reasons)

            selected = row_index < pick_count
            if selected:
                selected_contaminated_rows += 1
                selected_contaminated_tasks.add(task_id)

            if len(examples) < 10:
                issue_ref = row.get("issue_ref") or {}
                examples.append(
                    {
                        "task_id": task_id,
                        "selected": selected,
                        "row_index": row_index,
                        "unique_id": row.get("unique_id", ""),
                        "issue": issue_ref.get("issue_text", "")
                        or row.get("issue_text", ""),
                        "learning": row.get("learning_text", ""),
                        "reasons": reasons,
                    }
                )

    return {
        "path": str(path),
        "tasks_total": len(data),
        "tasks_with_contaminated_base": len(contaminated_tasks),
        "contaminated_base_rows": contaminated_rows,
        "tasks_with_selected_contamination": len(selected_contaminated_tasks),
        "selected_contaminated_rows": selected_contaminated_rows,
        "reason_counts": dict(sorted(reason_counts.items())),
        "examples": examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit SQL memory bank and retrieval bases for contamination."
    )
    parser.add_argument("--memory-bank", default=DEFAULT_KB, help="KB JSON path")
    parser.add_argument(
        "--retrieval",
        action="append",
        default=[],
        help=(
            "Retrieval JSON file or directory. Can be passed multiple times; "
            "directories are searched recursively for knowledge_retrieval_bases.json."
        ),
    )
    parser.add_argument("--output", default="", help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report: Dict[str, Any] = {
        "memory_bank": audit_kb(Path(args.memory_bank)),
        "retrieval_files": [],
    }

    for path in iter_retrieval_paths(args.retrieval):
        report["retrieval_files"].append(audit_retrieval_file(path))

    output = json.dumps(report, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
