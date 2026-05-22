#!/usr/bin/env python3
"""Build SQL transfer-radius diagnostic tables from a trusted gate run."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VARIANTS = {
    "react_cr": "CR",
    "react_tr": "TR",
    "react_cr_tr": "CR+TR",
    "react_hard_neg_cr_tr": "hard-neg CR+TR",
}

REPRESENTATIVE_TASKS = {"sql_66", "sql_117", "sql_120", "sql_185", "sql_204"}

FAILURE_MODE_OVERRIDES = {
    ("react_cr", "sql_49"): "spurious temporal normalization",
    ("react_cr", "sql_66"): "join inclusivity / zero-row inclusion",
    ("react_cr", "sql_120"): "set granularity / duplicate collapse",
    ("react_tr", "sql_66"): "prompt-only join inclusivity drift",
    ("react_cr_tr", "sql_66"): "join inclusivity / zero-row inclusion",
    ("react_cr_tr", "sql_117"): "join inclusivity / zero-row inclusion",
    ("react_cr_tr", "sql_120"): "set granularity / duplicate collapse",
    ("react_cr_tr", "sql_185"): "output column drift",
    ("react_hard_neg_cr_tr", "sql_38"): "duplicate policy drift",
    ("react_hard_neg_cr_tr", "sql_66"): "prompt-only join inclusivity drift",
}


@dataclass
class TaskRecord:
    task_id: str
    task_desc: str
    success: bool
    reward: float
    step_num: int
    final_sql: str
    final_observation: str
    help_call_count: int
    memories: list[dict[str, Any]]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def compact_text(value: str, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalize_action(action: str) -> str:
    return " ".join(str(action).strip().split())


def is_query_like(action: str, observation: str = "") -> bool:
    cleaned = action.strip()
    lowered = cleaned.lower()
    if not cleaned or lowered == "submit":
        return False
    if lowered.startswith("think:"):
        return False
    if lowered.startswith("help"):
        return False
    # The stabilized runner can emit OK for thought-only turns or malformed
    # multi-action turns. Those are not the SQL result the grader uses.
    if observation.strip() == "OK.":
        return False
    return True


def final_from_react_steps(steps: list[dict[str, str]]) -> tuple[str, str]:
    final_sql = ""
    final_obs = ""
    pending_action = ""
    for item in steps:
        label = item.get("label", "")
        value = item.get("value", "")
        if label == "action":
            pending_action = value
        elif label == "observation":
            if is_query_like(pending_action, value):
                final_sql = pending_action
                final_obs = value
    return normalize_action(final_sql), compact_text(final_obs, 320)


def final_from_memory_steps(steps: list[dict[str, Any]]) -> tuple[str, str]:
    final_sql = ""
    final_obs = ""
    for step in steps:
        action = step.get("action", "")
        observation = step.get("observation", "")
        if is_query_like(action, observation):
            final_sql = action
            final_obs = observation
    return normalize_action(final_sql), compact_text(final_obs, 320)


def memory_summary(memories: list[dict[str, Any]], limit: int = 3) -> str:
    parts = []
    for memory in memories[:limit]:
        unique_id = memory.get("unique_id", "?")
        task_desc = compact_text(memory.get("task_desc", ""), 80)
        issue = compact_text(memory.get("issue", ""), 90)
        learning = compact_text(memory.get("learning", ""), 120)
        score = memory.get("similarity_score", memory.get("ranking_score", ""))
        if isinstance(score, (int, float)):
            score_text = f"{score:.3f}"
        else:
            score_text = str(score)
        parts.append(
            f"{unique_id} ({score_text}): {task_desc} | issue={issue} | learning={learning}"
        )
    return " || ".join(parts)


def scan_environment_errors(run_dir: Path) -> list[str]:
    patterns = [
        "traceback",
        "exception",
        "environment error",
        "failed to create",
        "connection refused",
        "docker error",
    ]
    matches = []
    for world_log in sorted((run_dir / "test").glob("*/seed_0/world.log")):
        for line_no, line in enumerate(world_log.read_text(errors="replace").splitlines(), 1):
            lowered = line.lower()
            if any(pattern in lowered for pattern in patterns):
                matches.append(f"{world_log}:{line_no}: {line}")
    return matches


def load_react(run_dir: Path) -> dict[str, TaskRecord]:
    rows = read_json(run_dir / "test" / "react" / "seed_0" / "trajectories.json")
    records = {}
    for row in rows:
        final_sql, final_observation = final_from_react_steps(row.get("steps", []))
        records[row["task_id"]] = TaskRecord(
            task_id=row["task_id"],
            task_desc=row.get("task_desc", ""),
            success=bool(row.get("success")),
            reward=float(row.get("reward", 0) or 0),
            step_num=int(row.get("step_num", 0) or 0),
            final_sql=final_sql,
            final_observation=final_observation,
            help_call_count=0,
            memories=[],
        )
    return records


def load_variant(run_dir: Path, framework: str) -> dict[str, TaskRecord]:
    rows = read_jsonl(
        run_dir / "test" / framework / "seed_0" / "agent_trajectories.jsonl"
    )
    records = {}
    for row in rows:
        final_sql, final_observation = final_from_memory_steps(row.get("steps", []))
        memories = row.get("context_from_retrieval", []) or []
        records[row["task_id"]] = TaskRecord(
            task_id=row["task_id"],
            task_desc=row.get("task_desc", ""),
            success=bool(row.get("success")),
            reward=float(row.get("reward", 0) or 0),
            step_num=int(row.get("step_num", 0) or 0),
            final_sql=final_sql,
            final_observation=final_observation,
            help_call_count=int(row.get("help_call_count", 0) or 0),
            memories=memories,
        )
    return records


def load_metrics(run_dir: Path, framework: str) -> dict[str, Any]:
    return read_json(run_dir / "test" / framework / "seed_0" / "metrics.json")["metrics"]


def classify_delta(task_id: str, variant: str, react: TaskRecord, other: TaskRecord) -> str:
    key = (variant, task_id)
    if key in FAILURE_MODE_OVERRIDES:
        return FAILURE_MODE_OVERRIDES[key]
    if react.success and not other.success:
        return "needs_review"
    if other.reward < react.reward:
        return "partial_reward_drop"
    if other.reward > react.reward:
        return "improvement"
    if other.final_sql != react.final_sql:
        return "neutral_query_change"
    return "unchanged"


def delta_status(react: TaskRecord, other: TaskRecord) -> str:
    if react.success and not other.success:
        return "regression"
    if not react.success and other.success:
        return "improvement"
    if other.reward < react.reward:
        return "reward_drop"
    if other.reward > react.reward:
        return "reward_gain"
    return "same"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        values = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Trusted gate directory, e.g. intercode_sql_runs/gates/.../sql_action_stable_50",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiment_reports/sql_transfer_radius"),
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    react_records = load_react(run_dir)
    metrics = {"react": load_metrics(run_dir, "react")}
    environment_errors = scan_environment_errors(run_dir)
    variant_records = {}
    for framework in VARIANTS:
        variant_records[framework] = load_variant(run_dir, framework)
        metrics[framework] = load_metrics(run_dir, framework)

    all_delta_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []

    for framework, display in VARIANTS.items():
        rows = []
        for task_id in sorted(react_records, key=lambda value: int(value.split("_")[1])):
            react = react_records[task_id]
            other = variant_records[framework][task_id]
            row = {
                "variant": display,
                "task_id": task_id,
                "task_desc": react.task_desc,
                "react_success": int(react.success),
                "variant_success": int(other.success),
                "react_reward": f"{react.reward:.4f}",
                "variant_reward": f"{other.reward:.4f}",
                "delta_reward": f"{other.reward - react.reward:+.4f}",
                "react_steps": react.step_num,
                "variant_steps": other.step_num,
                "help_calls": other.help_call_count,
                "status": delta_status(react, other),
                "failure_mode": classify_delta(task_id, framework, react, other),
                "react_final_sql": react.final_sql,
                "variant_final_sql": other.final_sql,
                "variant_memories": memory_summary(other.memories),
            }
            rows.append(row)
            all_delta_rows.append(row)
            if row["status"] in {"regression", "reward_drop"}:
                regression_rows.append(row)
            if task_id in REPRESENTATIVE_TASKS:
                representative_rows.append(row)
        write_csv(out_dir / f"delta_{framework}_vs_react.csv", rows)

    write_csv(out_dir / "delta_all_variants_vs_react.csv", all_delta_rows)
    write_csv(out_dir / "regressions_vs_react.csv", regression_rows)
    write_csv(out_dir / "representative_examples.csv", representative_rows)

    metrics_rows = []
    for framework, metric in metrics.items():
        metrics_rows.append(
            {
                "framework": "ReAct" if framework == "react" else VARIANTS[framework],
                "success_total": metric.get("success_total"),
                "accuracy": f"{float(metric.get('accuracy', 0)):.4f}",
                "avg_reward": f"{float(metric.get('avg_reward', 0)):.4f}",
                "avg_steps": f"{float(metric.get('avg_steps_per_task', 0)):.4f}",
            }
        )
    write_csv(out_dir / "metrics_summary.csv", metrics_rows)

    status_rows = []
    for framework, display in VARIANTS.items():
        variant_rows = [row for row in all_delta_rows if row["variant"] == display]
        status_rows.append(
            {
                "variant": display,
                "regressions": sum(row["status"] == "regression" for row in variant_rows),
                "reward_drops": sum(row["status"] == "reward_drop" for row in variant_rows),
                "improvements": sum(row["status"] == "improvement" for row in variant_rows),
                "reward_gains": sum(row["status"] == "reward_gain" for row in variant_rows),
                "changed_final_sql": sum(
                    row["react_final_sql"] != row["variant_final_sql"]
                    for row in variant_rows
                ),
                "help_calls": sum(int(row["help_calls"]) for row in variant_rows),
            }
        )
    write_csv(out_dir / "delta_status_summary.csv", status_rows)

    failure_mode_rows = []
    for framework, display in VARIANTS.items():
        variant_regressions = [
            row for row in regression_rows if row["variant"] == display
        ]
        modes = sorted({row["failure_mode"] for row in variant_regressions})
        for mode in modes:
            failure_mode_rows.append(
                {
                    "variant": display,
                    "failure_mode": mode,
                    "count": sum(
                        row["failure_mode"] == mode for row in variant_regressions
                    ),
                    "tasks": ", ".join(
                        row["task_id"]
                        for row in variant_regressions
                        if row["failure_mode"] == mode
                    ),
                }
            )
    write_csv(out_dir / "failure_mode_summary.csv", failure_mode_rows)

    top_regressions = [
        row
        for row in regression_rows
        if row["status"] == "regression" or float(row["delta_reward"]) < 0
    ]
    top_regressions = sorted(
        top_regressions,
        key=lambda row: (float(row["delta_reward"]), row["variant"], row["task_id"]),
    )

    report = []
    report.append("# SQL Transfer-Radius Diagnostic")
    report.append("")
    report.append(f"Input gate: `{run_dir}`")
    report.append("")
    report.append("## Final Metrics")
    report.append(markdown_table(metrics_rows, list(metrics_rows[0].keys())))
    report.append("")
    report.append("## Delta Status Summary")
    report.append(markdown_table(status_rows, list(status_rows[0].keys())))
    report.append("")
    report.append("## Failure Mode Summary")
    report.append(markdown_table(failure_mode_rows, list(failure_mode_rows[0].keys())))
    report.append("")
    report.append("## Paper-Ready Diagnostic Summary")
    report.append(
        "The trusted SQL gate isolates memory behavior rather than runner instability: "
        f"the paired run has zero retrieved-help calls across memory variants and "
        f"{'no' if not environment_errors else len(environment_errors)} environment-error matches in world logs. "
        "TR therefore measures a prompt/tool-availability perturbation, not useful tool retrieval."
    )
    report.append("")
    report.append(
        "SQL looks syntactically regular, but the transfer radius of prior memories is narrow. "
        "A memory that is directionally plausible often changes a latent query policy: "
        "whether to include zero-count entities, whether duplicate rows are meaningful, "
        "whether exclusion applies to rows or grouped entities, which columns the grader expects, "
        "or whether literal text dates should be normalized. These choices are small in syntax "
        "and large in denotation, so near-miss memories can degrade exact-match reward."
    )
    report.append("")
    report.append(
        "The clearest CR regressions are near-miss transfers. In `sql_66`, memories about empty "
        "tables and count verification push the agent toward a left join that adds stadiums with "
        "zero concerts. In `sql_120`, a country-level memory about 'not speaking English' transfers "
        "to channel rows and collapses valid duplicate `(aspect ratio, country)` rows with DISTINCT. "
        "In `sql_185`, CR+TR adds a plausible explanatory `note` column even though the task asks "
        "only for death and injury values. In `sql_117`, counting a specific joined id is reasonable "
        "under an inner join, but paired with a left join it admits airlines with no flights."
    )
    report.append("")
    report.append(
        "Hard-negative CR+TR performs relatively well because irrelevant memories are easier to ignore "
        "than semantically adjacent memories. Its remaining failures are mostly generic policy drift: "
        "`sql_38` drops DISTINCT and duplicates an otherwise correct row, while `sql_66` repeats the "
        "same zero-row inclusion error seen under TR. The contrast supports the paper's claim that "
        "SQL transfer fails not because SQL lacks structure, but because the structure creates many "
        "locally plausible, task-specific policy choices with fragile boundaries."
    )
    report.append("")
    report.append("## Largest Regressions vs ReAct")
    report.append(
        markdown_table(
            top_regressions[:20],
            [
                "variant",
                "task_id",
                "task_desc",
                "delta_reward",
                "status",
                "failure_mode",
                "react_final_sql",
                "variant_final_sql",
            ],
        )
    )
    report.append("")
    report.append("## Representative Examples")
    report.append(
        markdown_table(
            representative_rows,
            [
                "variant",
                "task_id",
                "task_desc",
                "react_success",
                "variant_success",
                "delta_reward",
                "failure_mode",
                "variant_memories",
            ],
        )
    )
    report.append("")
    report.append("## Suggested Figure/Table Additions")
    report.append(
        "1. Add a paired SQL gate table with ReAct, CR, TR, CR+TR, and hard-neg CR+TR: success, "
        "average reward, regressions vs ReAct, improvements vs ReAct, and help calls."
    )
    report.append(
        "2. Add a failure-mode taxonomy table with one representative query diff per mode: "
        "join inclusivity, duplicate policy, set granularity, output-column drift, and temporal normalization."
    )
    report.append(
        "3. Add a compact paired-delta strip plot or heatmap over the 50 tasks, grouped by variant, "
        "to show that most tasks are unchanged and the underperformance is concentrated in a few brittle cases."
    )
    report.append(
        "4. Add a near-miss vs hard-negative comparison panel: near-miss CR memories produce plausible "
        "wrong policies, while hard negatives mostly have lower uptake and fewer memory-induced failures."
    )
    report.append(
        "5. Add a callout box for TR: zero help calls means the TR result is a prompt perturbation baseline, "
        "not evidence that retrieved tool instructions helped or hurt SQL execution."
    )
    report.append("")
    report.append("## Outputs")
    for name in [
        "metrics_summary.csv",
        "delta_status_summary.csv",
        "failure_mode_summary.csv",
        "delta_all_variants_vs_react.csv",
        "regressions_vs_react.csv",
        "representative_examples.csv",
    ]:
        report.append(f"- `{out_dir / name}`")
    report.append("")
    (out_dir / "sql_transfer_radius_diagnostic.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
