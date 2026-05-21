#!/usr/bin/env python3
"""Audit and optionally sanitize a WebShop memory-retrieval knowledge base.

The audit is intentionally lexical and reproducible: it does not call models or
embeddings. It flags memories that can push WebShop agents toward stale protocol
actions, over-strict rejection behavior, unavailable-product conclusions, or
product-specific leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_EXCLUDE_REASONS = (
    "stale_protocol_finish",
    "contradictory_unavailable",
    "over_strict_unavailable",
    "environment_navigation_bug",
)

TEXT_FIELDS = (
    "task_desc",
    "obj_type",
    "verbs",
    "goal_phase",
    "issue_text",
    "learning_text",
)

ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
FINISH_RE = re.compile(r"\bfinish\s*(?:\[|\(|$|\s)", re.IGNORECASE)
UNAVAILABLE_RE = re.compile(
    r"\b("
    r"unavailable|not available|no (?:matching |suitable |relevant )?(?:product|item|match|option)s?|"
    r"cannot find|can't find|could not find|unable to find|not find|"
    r"inaccessible|uninspectable|failed to navigate|navigation (?:failed|fails)|"
    r"returns? to search|environment (?:issue|bug)|system error"
    r")\b",
    re.IGNORECASE,
)
STRICT_RE = re.compile(
    r"\b("
    r"always|never|must|only|strictly|reject|discard|do not|don't|"
    r"perfect(?:ly)?|exact(?:ly)?|explicit(?:ly)?|mandatory|non-negotiable|"
    r"all required|all specified|violating"
    r")\b",
    re.IGNORECASE,
)
BUY_OR_COMPROMISE_RE = re.compile(
    r"\b("
    r"buy|purchase|buy now|accept|assume|infer|approximate|substitute|compromise|"
    r"reasonable substitute|make an informed assumption"
    r")\b",
    re.IGNORECASE,
)
PRODUCT_SPECIFIC_RE = re.compile(
    r"\b("
    r"(?:selected|clicked|chose|bought|rejected)\s+product\s+(?:id\s*)?[A-Z0-9]{8,}|"
    r"product\s+B0[A-Z0-9]{8}"
    r")\b",
    re.IGNORECASE,
)
PROTOCOL_ACTION_RE = re.compile(r"\b(?:click|search|help|think)\s*\[", re.IGNORECASE)


def load_kb(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of KB entries in {path}")
    return payload


def entry_text(entry: Dict[str, Any]) -> str:
    parts = []
    for field in TEXT_FIELDS:
        value = entry.get(field, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        parts.append(str(value or ""))
    return "\n".join(parts)


def flag_entry(entry: Dict[str, Any]) -> List[str]:
    text = entry_text(entry)
    reasons: List[str] = []

    if FINISH_RE.search(text):
        reasons.append("stale_protocol_finish")

    has_unavailable = bool(UNAVAILABLE_RE.search(text))
    has_strict = bool(STRICT_RE.search(text))
    has_buy_or_compromise = bool(BUY_OR_COMPROMISE_RE.search(text))

    if has_unavailable and has_buy_or_compromise:
        reasons.append("contradictory_unavailable")
    if has_unavailable and has_strict:
        reasons.append("over_strict_unavailable")
    if re.search(r"\b(environment|system)\s+(?:issue|bug|error)\b", text, re.IGNORECASE):
        reasons.append("environment_navigation_bug")
    if ASIN_RE.search(text) or PRODUCT_SPECIFIC_RE.search(text):
        reasons.append("product_specific")
    if PROTOCOL_ACTION_RE.search(text):
        reasons.append("protocol_action_literal")

    return reasons


def compact_entry(entry: Dict[str, Any], reasons: Sequence[str]) -> Dict[str, Any]:
    return {
        "unique_id": entry.get("unique_id", ""),
        "task_desc": entry.get("task_desc", ""),
        "valid_level": entry.get("valid_level", ""),
        "goal_phase": entry.get("goal_phase", ""),
        "issue_text": entry.get("issue_text", ""),
        "learning_text": entry.get("learning_text", ""),
        "reasons": list(reasons),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unique_id",
        "valid_level",
        "goal_phase",
        "reasons",
        "issue_text",
        "learning_text",
        "task_desc",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "unique_id": row.get("unique_id", ""),
                "valid_level": row.get("valid_level", ""),
                "goal_phase": row.get("goal_phase", ""),
                "reasons": ",".join(row.get("reasons", [])),
                "issue_text": row.get("issue_text", ""),
                "learning_text": row.get("learning_text", ""),
                "task_desc": row.get("task_desc", ""),
            })


def parse_reasons(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and optionally sanitize a WebShop knowledge_base.json"
    )
    parser.add_argument(
        "--memory-bank",
        required=True,
        type=Path,
        help="Input WebShop knowledge_base.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional CSV report path for flagged entries",
    )
    parser.add_argument(
        "--write-sanitized",
        type=Path,
        default=None,
        help="Optional path for a sanitized knowledge_base.json",
    )
    parser.add_argument(
        "--exclude-reasons",
        default=",".join(DEFAULT_EXCLUDE_REASONS),
        help=(
            "Comma-separated flag reasons to remove when --write-sanitized is set "
            f"(default: {','.join(DEFAULT_EXCLUDE_REASONS)})"
        ),
    )
    parser.add_argument(
        "--exclude-product-specific",
        action="store_true",
        help="Also remove entries flagged as product_specific.",
    )
    args = parser.parse_args()

    kb = load_kb(args.memory_bank)
    flagged: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    exclude_reasons = parse_reasons(args.exclude_reasons)
    if args.exclude_product_specific:
        exclude_reasons.add("product_specific")

    sanitized: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for entry in kb:
        reasons = flag_entry(entry)
        reason_counts.update(reasons)
        if reasons:
            flagged.append(compact_entry(entry, reasons))

        if exclude_reasons.intersection(reasons):
            removed.append(compact_entry(entry, reasons))
        else:
            sanitized.append(entry)

    summary = {
        "memory_bank": str(args.memory_bank),
        "total_entries": len(kb),
        "flagged_entries": len(flagged),
        "reason_counts": dict(sorted(reason_counts.items())),
        "exclude_reasons": sorted(exclude_reasons),
        "sanitized_entries": len(sanitized) if args.write_sanitized else None,
        "removed_entries": len(removed) if args.write_sanitized else None,
    }
    report = {
        "summary": summary,
        "flagged": flagged,
        "removed": removed if args.write_sanitized else [],
    }

    print(json.dumps(summary, indent=2))

    if args.report_json:
        write_json(args.report_json, report)
    if args.report_csv:
        write_csv(args.report_csv, flagged)
    if args.write_sanitized:
        write_json(args.write_sanitized, sanitized)
        provenance_path = args.write_sanitized.with_suffix(".audit.json")
        write_json(provenance_path, report)
        print(f"Wrote sanitized KB: {args.write_sanitized}")
        print(f"Wrote audit provenance: {provenance_path}")


if __name__ == "__main__":
    main()
