#!/usr/bin/env python3
"""Sanitize InterCode SQL memory banks.

The SQL environment uses ``submit`` as a runner control action: execute SQL
queries until the current observation is the final result, then issue exactly
``submit``. Memories that teach answer-after-submit or other runner protocol
details contaminate CR/TR retrieval, so this utility quarantines them while
keeping SQL reasoning learnings.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT = (
    "intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json"
)
DEFAULT_OUTPUT = (
    "intercode_sql_runs/memory_agent_runs/train/react_reflexion/"
    "knowledge_base.sql_sanitized.json"
)

SQL_COMPLETENESS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bcomplete sql\b",
        r"\bcomplete sql quer(?:y|ies)\b",
        r"\bcomplete select\b",
        r"\bselect statements? (?:are|is)? ?complete\b",
        r"\bcomplete cte\b",
        r"\bmain query\b",
        r"\bfrom clause\b",
        r"\bjoin clauses?\b",
        r"\bconstruct complete\b",
        r"\bprovided after the select keyword\b",
    ]
]

CONTAMINATION_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = [
    (
        "submit_answer_protocol",
        re.compile(
            r"\bsubmit(?:ted|ting)?\b.{0,80}\b("
            r"final answer|answer|text|line|command|action|keyword|special agent|"
            r"natural language|descriptive|explanation|empty string|empty result|"
            r"nothing after|without arguments|directly"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "answer_after_submit_protocol",
        re.compile(
            r"\b(final answer|answer|text|natural language|descriptive|explanation)"
            r"\b.{0,80}\bsubmit(?:ted|ting)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "non_sql_answer_protocol",
        re.compile(
            r"\b(natural language|descriptive|textual explanation|string as an answer)"
            r"\b.{0,80}\b(submit|answer|final|explanation)\b|"
            r"\b(submit|answer|final)\b.{0,80}\b"
            r"(natural language|descriptive|textual explanation|string as an answer|"
            r"not a sql command|not an sql query|not as part of a sql statement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "final_answer_protocol",
        re.compile(r"\bfinal answer\b|\bfinal answer value\b", re.IGNORECASE),
    ),
    (
        "submit_colon_empty_protocol",
        re.compile(r"\bsubmit:\b|\bnothing after\b|\bempty string\b", re.IGNORECASE),
    ),
    (
        "runner_evaluation_artifact",
        re.compile(
            r"\b(system marked|marked as incorrect|evaluation system|no sql correction needed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "data_insufficiency_answer_protocol",
        re.compile(
            r"\b(conclude data insufficiency|insufficient data)\b.{0,80}\b"
            r"(submit|answer|message|statement)\b",
            re.IGNORECASE,
        ),
    ),
]


def _entry_text(entry: Dict[str, Any]) -> str:
    fields = [
        "issue_text",
        "learning_text",
        "obj_type",
        "verbs",
        "goal_phase",
        "task_desc",
    ]
    return "\n".join(str(entry.get(field, "") or "") for field in fields)


def classify_memory(entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return ``(should_quarantine, reasons)`` for one KB entry."""
    text = _entry_text(entry)
    issue_learning = "\n".join(
        str(entry.get(field, "") or "") for field in ["issue_text", "learning_text"]
    )

    # Keep memories whose "submit" mention is really a SQL-completeness lesson.
    if any(pattern.search(issue_learning) for pattern in SQL_COMPLETENESS_PATTERNS):
        return False, []

    reasons = [
        reason
        for reason, pattern in CONTAMINATION_PATTERNS
        if pattern.search(text)
    ]
    return bool(reasons), reasons


def split_memory_bank(
    memory_bank: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    kept: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()

    for index, entry in enumerate(memory_bank):
        quarantine, reasons = classify_memory(entry)
        if quarantine:
            item = dict(entry)
            item["_sanitizer"] = {
                "source_index": index,
                "reasons": reasons,
            }
            quarantined.append(item)
            reason_counts.update(reasons)
        else:
            kept.append(entry)

    return kept, quarantined, reason_counts


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def sanitize_memory_bank(input_path: Path, output_path: Path) -> Dict[str, Any]:
    with input_path.open("r") as f:
        memory_bank = json.load(f)

    if not isinstance(memory_bank, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    kept, quarantined, reason_counts = split_memory_bank(memory_bank)
    quarantine_path = output_path.with_name(f"{output_path.stem}.quarantine.json")
    report_path = output_path.with_name(f"{output_path.stem}.report.json")

    write_json(output_path, kept)
    write_json(quarantine_path, quarantined)

    report = {
        "created_at": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "quarantine_path": str(quarantine_path),
        "total_memories": len(memory_bank),
        "kept_memories": len(kept),
        "quarantined_memories": len(quarantined),
        "reason_counts": dict(sorted(reason_counts.items())),
        "quarantined_unique_ids": [
            item.get("unique_id", "") for item in quarantined
        ],
    }
    write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a sanitized InterCode SQL memory bank."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source KB JSON")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Sanitized KB JSON to write",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = sanitize_memory_bank(Path(args.input), Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
