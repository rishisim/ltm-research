#!/usr/bin/env python3
"""Generate Phase 4 final tables, figures, and summary report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_ORDER = ["alfworld_trials7_gemini", "sql_trials7_gemini", "scienceworld_trials7_gemini"]
ENV_LABELS = {
    "alfworld_trials7_gemini": "ALFWorld",
    "sql_trials7_gemini": "SQL",
    "scienceworld_trials7_gemini": "ScienceWorld",
}
FRAMEWORK_ORDER = ["react", "react_cr", "react_tr", "react_cr_tr", "react_hard_neg_cr_tr"]
FRAMEWORK_LABELS = {
    "react": "ReAct",
    "react_cr": "ReAct + CR",
    "react_tr": "ReAct + TR",
    "react_cr_tr": "ReAct + CR + TR",
    "react_hard_neg_cr_tr": "Hard-neg CR + TR",
}
FRAMEWORK_COLORS = {
    "react_cr": "#4C78A8",
    "react_tr": "#F58518",
    "react_cr_tr": "#54A24B",
    "react_hard_neg_cr_tr": "#B279A2",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def fnum(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def tex_escape(text: Any) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in out)


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return r"\textemdash{}"
    return f"{value:.{digits}f}"


def fmt_percent(value: Optional[float], std: Optional[float] = None, bold: bool = False) -> str:
    if value is None:
        return r"\textemdash{}"
    text = f"{value * 100:.1f}"
    if std is not None:
        text += rf" $\pm$ {std * 100:.1f}"
    if bold:
        text = rf"\textbf{{{text}}}"
    return text


def fmt_mean_std(value: Optional[float], std: Optional[float] = None, digits: int = 2) -> str:
    if value is None:
        return r"\textemdash{}"
    text = f"{value:.{digits}f}"
    if std is not None:
        text += rf" $\pm$ {std:.{digits}f}"
    return text


def fmt_delta_pp(value: Optional[float], std: Optional[float] = None) -> str:
    if value is None:
        return r"\textemdash{}"
    prefix = "+" if value > 0 else ""
    text = f"{prefix}{value * 100:.1f}"
    if std is not None:
        text += rf" $\pm$ {std * 100:.1f}"
    return text


def ordered(rows: Iterable[Dict[str, str]], include_react: bool = True) -> List[Dict[str, str]]:
    by_key = {(row["run_group"], row["framework"]): row for row in rows}
    result: List[Dict[str, str]] = []
    for env in ENV_ORDER:
        for fw in FRAMEWORK_ORDER:
            if not include_react and fw == "react":
                continue
            row = by_key.get((env, fw))
            if row:
                result.append(row)
    return result


def rows_by_env(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {env: [] for env in ENV_ORDER}
    for row in rows:
        out.setdefault(row["run_group"], []).append(row)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def make_main_results_table(framework_rows: Sequence[Dict[str, str]], paired_rows: Sequence[Dict[str, str]]) -> str:
    paired = {(row["run_group"], row["framework"]): row for row in paired_rows}
    best_by_env: Dict[str, float] = {}
    for env, env_rows in rows_by_env(framework_rows).items():
        values = [fnum(row.get("success_rate_mean")) for row in env_rows]
        values = [value for value in values if value is not None]
        if values:
            best_by_env[env] = max(values)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Final held-out Stage C results using train-only trials7 KBs, Gemini Flash, Gemini embeddings, \texttt{VALID\_NEXT\_TRIAL}, and \texttt{max\_learnings=5}. Values are mean $\pm$ std over three seeds.}",
        r"\label{tab:final-main-results}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Environment & Framework & Tasks & Success (\%) & Score & $\Delta$ success (pp) \\",
        r"\midrule",
    ]
    last_env = ""
    for row in ordered(framework_rows):
        env = row["run_group"]
        if last_env and env != last_env:
            lines.append(r"\midrule")
        last_env = env
        fw = row["framework"]
        success = fnum(row.get("success_rate_mean"))
        success_std = fnum(row.get("success_rate_std"))
        score = fnum(row.get("avg_reward_mean"))
        score_std = fnum(row.get("avg_reward_std"))
        if score is None:
            score = success
            score_std = success_std
        task_count = fnum(row.get("task_count_mean"))
        delta_row = paired.get((env, fw), {})
        delta = None if fw == "react" else fnum(delta_row.get("success_rate_delta_mean"))
        delta_std = None if fw == "react" else fnum(delta_row.get("success_rate_delta_std"))
        is_best = success is not None and abs(success - best_by_env.get(env, -1.0)) < 1e-12
        lines.append(
            " & ".join(
                [
                    tex_escape(ENV_LABELS[env]),
                    tex_escape(FRAMEWORK_LABELS[fw]),
                    fmt_float(task_count, 0),
                    fmt_percent(success, success_std, bold=is_best),
                    fmt_mean_std(score, score_std, 3),
                    r"\textemdash{}" if fw == "react" else fmt_delta_pp(delta, delta_std),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"{\footnotesize Score is average reward when logged; for ALFWorld ReAct, success rate is used because reward was not emitted in the final record.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def kb_inventory_rows() -> List[Dict[str, str]]:
    alf = load_json(PROJECT_ROOT / "final_runs/kb/alfworld_train_reflexion_trials7/audit_summary.json")
    sql = load_json(PROJECT_ROOT / "final_runs/kb/sql_train_reflexion_trials7_sanitized/audit_summary.json")
    sw = load_json(PROJECT_ROOT / "final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat/audit_summary.json")
    return [
        {
            "env": "ALFWorld",
            "kb": "alfworld_train_reflexion_trials7",
            "kb_tex": r"\shortstack[l]{\texttt{alfworld\_train}\\\texttt{reflexion\_trials7}}",
            "coverage": f"{alf['coverage']['observed_unique_trajectory_task_ids']}/{alf['coverage']['intended_train_games']} observed train games",
            "max_trial": str(alf["promoted"]["max_ref_trial"]),
            "memories": str(alf["promoted"]["kb_entries"]),
            "vnt": str(alf["promoted"]["valid_level_counts"].get("VALID_NEXT_TRIAL", 0)),
            "audit": "Train-only refs; no eval overlap; original 120-game manifest absent.",
        },
        {
            "env": "SQL",
            "kb": "sql_train_reflexion_trials7_sanitized",
            "kb_tex": r"\shortstack[l]{\texttt{sql\_train\_reflexion}\\\texttt{trials7\_sanitized}}",
            "coverage": f"{sql['split_audit']['run_unique_task_ids']}/{sql['split_audit']['train_manifest_tasks']} train tasks",
            "max_trial": str(sql["trajectory_audit"]["max_trial"]),
            "memories": str(sql["kb_audit"]["sanitized_memories"]),
            "vnt": str(sql["kb_audit"]["sanitized_valid_level_counts"].get("VALID_NEXT_TRIAL", 0)),
            "audit": "Sanitized; no dev/test overlap; protocol memories quarantined.",
        },
        {
            "env": "ScienceWorld",
            "kb": "scienceworld_train_reflexion_trials7_va80_30cat",
            "kb_tex": r"\shortstack[l]{\texttt{scienceworld\_train}\\\texttt{reflexion\_trials7}\\\texttt{va80\_30cat}}",
            "coverage": f"{sw['coverage']['covered_categories']}/{sw['coverage']['intended_categories']} categories, {sw['coverage']['covered_variations']} variations",
            "max_trial": str(sw["trajectory_audit"]["max_trial"]),
            "memories": str(sw["kb_audit"]["total_memories"]),
            "vnt": str(sw["kb_audit"]["valid_next_trial_count"]),
            "audit": "Train-only; 1 variation/category; failed extractions repaired.",
        },
    ]


def make_kb_inventory_table() -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Final train-only KB inventory for the trials7 protocol. VNT = \texttt{VALID\_NEXT\_TRIAL}.}",
        r"\label{tab:final-kb-inventory}",
        r"\small",
        r"\begin{tabular}{@{}lp{0.20\linewidth}p{0.16\linewidth}cccp{0.22\linewidth}@{}}",
        r"\toprule",
        r"Environment & KB artifact & Coverage & Max trial & Memories & VNT & Audit note \\",
        r"\midrule",
    ]
    for row in kb_inventory_rows():
        lines.append(
            " & ".join(
                [
                    tex_escape(row["env"]),
                    row["kb_tex"],
                    tex_escape(row["coverage"]),
                    tex_escape(row["max_trial"]),
                    tex_escape(row["memories"]),
                    tex_escape(row["vnt"]),
                    tex_escape(row["audit"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def make_memory_usage_table(framework_rows: Sequence[Dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Final memory usage and trajectory cost summaries from raw trajectories. Values are mean $\pm$ std over seeds.}",
        r"\label{tab:final-memory-usage}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Environment & Framework & Context tasks & Retrieved/task & Help calls & Help/task & Steps/task \\",
        r"\midrule",
    ]
    last_env = ""
    for row in ordered(framework_rows):
        env = row["run_group"]
        if last_env and env != last_env:
            lines.append(r"\midrule")
        last_env = env
        lines.append(
            " & ".join(
                [
                    tex_escape(ENV_LABELS[env]),
                    tex_escape(FRAMEWORK_LABELS[row["framework"]]),
                    fmt_mean_std(fnum(row.get("context_tasks_mean")), fnum(row.get("context_tasks_std")), 1),
                    fmt_mean_std(
                        fnum(row.get("avg_retrieved_learnings_per_task_mean")),
                        fnum(row.get("avg_retrieved_learnings_per_task_std")),
                        2,
                    ),
                    fmt_mean_std(fnum(row.get("help_calls_total_mean")), fnum(row.get("help_calls_total_std")), 1),
                    fmt_mean_std(
                        fnum(row.get("avg_help_calls_per_task_mean")),
                        fnum(row.get("avg_help_calls_per_task_std")),
                        2,
                    ),
                    fmt_mean_std(
                        fnum(row.get("avg_steps_from_trajectories_mean")),
                        fnum(row.get("avg_steps_from_trajectories_std")),
                        2,
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def make_paired_deltas_table(paired_rows: Sequence[Dict[str, str]]) -> str:
    by_key = {(row["run_group"], row["framework"]): row for row in paired_rows}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired task-level deltas against ReAct on the final held-out Stage C runs. Values are mean $\pm$ std over seeds.}",
        r"\label{tab:final-paired-deltas}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Environment & Framework & Common tasks & $\Delta$ success (pp) & $\Delta$ score & Improved & Regressed \\",
        r"\midrule",
    ]
    last_env = ""
    for env in ENV_ORDER:
        for fw in FRAMEWORK_ORDER:
            if fw == "react":
                continue
            row = by_key.get((env, fw))
            if not row:
                continue
            if last_env and env != last_env:
                lines.append(r"\midrule")
            last_env = env
            reward_delta = fnum(row.get("avg_reward_delta_mean"))
            reward_delta_std = fnum(row.get("avg_reward_delta_std"))
            lines.append(
                " & ".join(
                    [
                        tex_escape(ENV_LABELS[env]),
                        tex_escape(FRAMEWORK_LABELS[fw]),
                        fmt_mean_std(fnum(row.get("common_tasks_mean")), fnum(row.get("common_tasks_std")), 0),
                        fmt_delta_pp(fnum(row.get("success_rate_delta_mean")), fnum(row.get("success_rate_delta_std"))),
                        fmt_mean_std(reward_delta, reward_delta_std, 3),
                        fmt_mean_std(fnum(row.get("improved_tasks_mean")), fnum(row.get("improved_tasks_std")), 1),
                        fmt_mean_std(fnum(row.get("regressed_tasks_mean")), fnum(row.get("regressed_tasks_std")), 1),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_gain_over_react(paired_rows: Sequence[Dict[str, str]], output: Path) -> None:
    setup_matplotlib()
    x = list(range(len(ENV_ORDER)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, fw in enumerate(FRAMEWORK_ORDER[1:]):
        values = []
        errors = []
        for env in ENV_ORDER:
            row = next((r for r in paired_rows if r["run_group"] == env and r["framework"] == fw), {})
            values.append((fnum(row.get("success_rate_delta_mean")) or 0.0) * 100)
            errors.append((fnum(row.get("success_rate_delta_std")) or 0.0) * 100)
        offsets = [pos + (i - 1.5) * width for pos in x]
        ax.bar(offsets, values, width=width, yerr=errors, color=FRAMEWORK_COLORS[fw], label=FRAMEWORK_LABELS[fw])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [ENV_LABELS[env] for env in ENV_ORDER])
    ax.set_ylabel("Success delta vs ReAct (pp)")
    ax.set_title("Final gain over ReAct")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_reward_vs_cost(framework_rows: Sequence[Dict[str, str]], output: Path) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    markers = {"alfworld_trials7_gemini": "o", "sql_trials7_gemini": "s", "scienceworld_trials7_gemini": "^"}
    for row in ordered(framework_rows):
        env = row["run_group"]
        fw = row["framework"]
        steps = fnum(row.get("avg_steps_from_trajectories_mean"))
        score = fnum(row.get("avg_reward_mean"))
        if score is None:
            score = fnum(row.get("success_rate_mean"))
        if steps is None or score is None:
            continue
        color = FRAMEWORK_COLORS.get(fw, "#555555")
        ax.scatter(steps, score, s=52, marker=markers[env], color=color, edgecolor="white", linewidth=0.7)
        ax.annotate(
            FRAMEWORK_LABELS[fw].replace("ReAct + ", "").replace("Hard-neg ", "HN "),
            (steps, score),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
            alpha=0.85,
        )
    handles = [
        plt.Line2D([0], [0], marker=markers[env], linestyle="", color="#555555", label=ENV_LABELS[env])
        for env in ENV_ORDER
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    ax.set_xlabel("Average steps per task (cost proxy)")
    ax.set_ylabel("Mean score")
    ax.set_title("Reward versus trajectory cost proxy")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_paired_delta_heatmap(paired_rows: Sequence[Dict[str, str]], output: Path) -> None:
    setup_matplotlib()
    frameworks = FRAMEWORK_ORDER[1:]
    matrix: List[List[float]] = []
    for env in ENV_ORDER:
        values: List[float] = []
        for fw in frameworks:
            row = next((r for r in paired_rows if r["run_group"] == env and r["framework"] == fw), {})
            values.append((fnum(row.get("success_rate_delta_mean")) or 0.0) * 100)
        matrix.append(values)

    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    vmax = max(abs(value) for row in matrix for value in row) or 1.0
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(frameworks)), [FRAMEWORK_LABELS[fw] for fw in frameworks], rotation=25, ha="right")
    ax.set_yticks(range(len(ENV_ORDER)), [ENV_LABELS[env] for env in ENV_ORDER])
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            color = "white" if abs(value) > vmax * 0.45 else "#222222"
            ax.text(x, y, f"{value:+.1f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Paired success delta vs ReAct (pp)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("pp")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_hard_negative_effect(framework_rows: Sequence[Dict[str, str]], output: Path) -> None:
    setup_matplotlib()
    by_key = {(row["run_group"], row["framework"]): row for row in framework_rows}
    values: List[float] = []
    reward_values: List[float] = []
    for env in ENV_ORDER:
        crtr = by_key[(env, "react_cr_tr")]
        hard = by_key[(env, "react_hard_neg_cr_tr")]
        values.append(((fnum(hard.get("success_rate_mean")) or 0.0) - (fnum(crtr.get("success_rate_mean")) or 0.0)) * 100)
        hard_score = fnum(hard.get("avg_reward_mean")) or fnum(hard.get("success_rate_mean")) or 0.0
        crtr_score = fnum(crtr.get("avg_reward_mean")) or fnum(crtr.get("success_rate_mean")) or 0.0
        reward_values.append(hard_score - crtr_score)

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = list(range(len(ENV_ORDER)))
    bars = ax.bar(x, values, color=["#4C78A8", "#F58518", "#54A24B"], width=0.55)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [ENV_LABELS[env] for env in ENV_ORDER])
    ax.set_ylabel("Hard-neg minus CR+TR success (pp)")
    ax.set_title("Hard-negative retrieval effect")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, value, reward_delta in zip(bars, values, reward_values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.7 if value >= 0 else -0.7
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f} pp\nscore {reward_delta:+.3f}",
            ha="center",
            va=va,
            fontsize=7.5,
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def headline_markdown(framework_rows: Sequence[Dict[str, str]], paired_rows: Sequence[Dict[str, str]]) -> str:
    paired = {(row["run_group"], row["framework"]): row for row in paired_rows}
    lines = [
        "| Environment | Best framework | Best success | ReAct success | Best delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for env in ENV_ORDER:
        env_rows = [row for row in framework_rows if row["run_group"] == env]
        best = max(env_rows, key=lambda row: fnum(row.get("success_rate_mean")) or -1.0)
        react = next(row for row in env_rows if row["framework"] == "react")
        best_fw = best["framework"]
        delta = 0.0 if best_fw == "react" else fnum(paired.get((env, best_fw), {}).get("success_rate_delta_mean")) or 0.0
        lines.append(
            f"| {ENV_LABELS[env]} | {FRAMEWORK_LABELS[best_fw]} | "
            f"{(fnum(best.get('success_rate_mean')) or 0.0) * 100:.1f}% | "
            f"{(fnum(react.get('success_rate_mean')) or 0.0) * 100:.1f}% | {delta * 100:+.1f} pp |"
        )
    return "\n".join(lines)


def make_summary(
    args: argparse.Namespace,
    framework_rows: Sequence[Dict[str, str]],
    paired_rows: Sequence[Dict[str, str]],
    aggregate_report: Dict[str, Any],
    stage_c_audit: Dict[str, Any],
) -> str:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    counts = aggregate_report.get("counts", {})
    lines = [
        "# Final Results Summary",
        "",
        "## Provenance",
        "",
        f"- Branch/worktree: `final/tables-figures` at `{PROJECT_ROOT}`",
        f"- Git commit at generation time: `{git_commit()}`",
        f"- Aggregation command: `{aggregate_report.get('audit', {}).get('command', '')}`",
        f"- Table/figure command: `{command}`",
        "- Inputs: `final_runs/eval/*_trials7_gemini`, `final_runs/kb/*trials7*`, `final_runs/audits/phase3_stage_C_audit.json`",
        "- Outputs: `paper/tables/final_*.tex`, `paper/figures/final_*.pdf`, `experiment_reports/final/aggregates/*`",
        "",
        "## Aggregate Counts",
        "",
        f"- Suite roots: {counts.get('suite_roots')}",
        f"- Per-seed rows: {counts.get('per_seed_rows')}",
        f"- Framework aggregate rows: {counts.get('framework_aggregate_rows')}",
        f"- Paired-delta rows: {counts.get('paired_delta_rows')}",
        f"- Stage C audit: {'PASS' if stage_c_audit.get('passed') else 'FAIL'} with {len(stage_c_audit.get('errors', []))} errors and {len(stage_c_audit.get('warnings', []))} warnings.",
        "",
        "## Headline Stage C Results",
        "",
        headline_markdown(framework_rows, paired_rows),
        "",
        "## Generated Tables",
        "",
        "- `paper/tables/final_main_results.tex`",
        "- `paper/tables/final_kb_inventory.tex`",
        "- `paper/tables/final_memory_usage.tex`",
        "- `paper/tables/final_paired_deltas.tex`",
        "",
        "## Generated Figures",
        "",
        "- `paper/figures/final_gain_over_react.pdf`",
        "- `paper/figures/final_reward_vs_cost.pdf`",
        "- `paper/figures/final_paired_delta_heatmap.pdf`",
        "- `paper/figures/final_hard_negative_effect.pdf`",
        "",
        "## Caveats",
        "",
        "- ALFWorld trials7 KB is train-only after quarantine, but the legacy exact 120-game train manifest is absent; audit observes 115 unique train trajectory task IDs and no valid_seen/valid_unseen overlap.",
        "- ScienceWorld final KB covers all 30 categories with the minimum accepted 1 train variation per category, not the preferred 3 variations per category.",
        "- Provider dollar cost is not available in final raw logs; `final_reward_vs_cost.pdf` uses average steps per task as a trajectory-cost proxy.",
        "- SQL `react_cr` task `sql_288` had recovered bare code-fence actions in seeds 0, 1, and 2; the environment rejected those actions and the task succeeded, so this is documented as non-fatal model-formatting noise.",
        "- Diagnostic ScienceWorld repair runs under `diagnostics/` are preserved but ignored by audit and aggregation discovery.",
        "- No retrieval-budget ablation was present under `final_runs/eval`, so the optional retrieval-budget plot was not generated.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final Phase 4 paper tables and figures")
    parser.add_argument("--aggregate-dir", type=Path, default=PROJECT_ROOT / "experiment_reports/final/aggregates")
    parser.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "paper/tables")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "paper/figures")
    parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "experiment_reports/final/final_results_summary.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate_dir = args.aggregate_dir.resolve()
    framework_rows = read_csv_rows(aggregate_dir / "final_run_framework_aggregate.csv")
    paired_rows = read_csv_rows(aggregate_dir / "final_run_paired_deltas_vs_react_aggregate.csv")
    aggregate_report = load_json(aggregate_dir / "aggregate_final_runs_report.json")
    stage_c_audit = load_json(PROJECT_ROOT / "final_runs/audits/phase3_stage_C_audit.json")

    write_text(args.tables_dir / "final_main_results.tex", make_main_results_table(framework_rows, paired_rows))
    write_text(args.tables_dir / "final_kb_inventory.tex", make_kb_inventory_table())
    write_text(args.tables_dir / "final_memory_usage.tex", make_memory_usage_table(framework_rows))
    write_text(args.tables_dir / "final_paired_deltas.tex", make_paired_deltas_table(paired_rows))

    save_gain_over_react(paired_rows, args.figures_dir / "final_gain_over_react.pdf")
    save_reward_vs_cost(framework_rows, args.figures_dir / "final_reward_vs_cost.pdf")
    save_paired_delta_heatmap(paired_rows, args.figures_dir / "final_paired_delta_heatmap.pdf")
    save_hard_negative_effect(framework_rows, args.figures_dir / "final_hard_negative_effect.pdf")

    write_text(args.summary_path, make_summary(args, framework_rows, paired_rows, aggregate_report, stage_c_audit))
    print(
        json.dumps(
            {
                "tables": 4,
                "figures": 4,
                "summary": str(args.summary_path),
                "stage_c_passed": bool(stage_c_audit.get("passed")),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
