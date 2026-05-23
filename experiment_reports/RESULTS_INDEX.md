# Results Index

Last updated: 2026-05-23

This index is a map of result artifacts currently preserved in the repository. It does not delete, move, or supersede any run output; use `archive/current_state_2026_05_23.md` for trust labels and caveats.

## Current trusted gates

| Area | Primary artifact | Backing run directory | Status |
| --- | --- | --- | --- |
| ScienceWorld dev-30, VA80, next-trial KB | `scienceworld_runs/memory_retrieval_v2/gates/dev_30_va80_next_trial_kb/summaries/summary_dev.md` | `scienceworld_runs/memory_retrieval_v2/gates/dev_30_va80_next_trial_kb` | Current broad ScienceWorld gate: 30 dev tasks, matched `max_valid_actions=80`, train-only Reflexion KB with next-trial filtering. |
| ScienceWorld dev-10 paired gate | `experiment_reports/scienceworld_dev_10_va80_next_trial_gate.md` | `scienceworld_runs/memory_retrieval_v2/gates/dev_10_va80_next_trial_kb` | Current small paired diagnostic with explicit paired deltas in `scienceworld_dev_10_va80_next_trial_paired_deltas.csv`. |
| SQL transfer-radius diagnostic | `experiment_reports/sql_transfer_radius/sql_transfer_radius_diagnostic.md` | `intercode_sql_runs/gates/gpt-5.4/sql_action_stable_50` | Trusted diagnostic for SQL memory transfer radius and failure modes. |
| ALFWorld multiseed summaries | `alfworld_runs/multiseed/gpt-5-mini/summaries/summary_all.csv`, `alfworld_runs/multiseed/gemini-2.5-flash/summaries/summary_all.csv` | `alfworld_runs/multiseed` | Preserved multiseed summaries; current files are single-framework summary slices rather than complete variant matrices. |
| InterCode SQL multiseed summaries | `intercode_sql_runs/multiseed/gpt-5-mini/summaries/summary_all.csv`, `intercode_sql_runs/multiseed/gemini-2.5-flash/summaries/summary_all.csv` | `intercode_sql_runs/multiseed` | Preserved multiseed summaries; current files are single-framework summary slices rather than complete variant matrices. |
| WebShop memory retrieval summaries | `webshop_runs/memory_retrieval_v2/memory_agent_runs/summaries/summary_all.csv` | `webshop_runs/memory_retrieval_v2/memory_agent_runs` | Preserved WebShop dev/test summary for hard-negative CR+TR. |

## ScienceWorld

| Artifact | What it contains | Notes |
| --- | --- | --- |
| `experiment_reports/scienceworld_dev_10_va80_next_trial_gate.md` | Human-readable dev-10 gate report for ReAct, CR, TR, CR+TR, and hard-neg CR+TR. | Matched `max_valid_actions=80`; retrieval filter uses `VALID_NEXT_TRIAL`. |
| `experiment_reports/scienceworld_dev_10_va80_next_trial_paired_deltas.csv` | Per-framework paired deltas vs ReAct for the dev-10 gate. | Use with the report above. |
| `scienceworld_runs/memory_retrieval_v2/gates/dev_10_va80_next_trial_kb/summaries/summary_dev.csv` | Machine-readable summary for the same dev-10 gate. | Backing run artifact. |
| `scienceworld_runs/memory_retrieval_v2/gates/dev_30_va80_next_trial_kb/summaries/summary_dev.md` | Broader dev-30 gate summary. | ReAct 8/30, CR 10/30, TR 11/30, CR+TR 9/30, hard-neg CR+TR 8/30. |
| `experiment_reports/scienceworld_dev_10_uncapped_gate.md` | Earlier uncapped dev-10 gate report. | Exploratory diagnostic only; KB lacked `VALID_NEXT_TRIAL` entries. |
| `scienceworld_runs/memory_retrieval_v2/kb_train_reflexion_trials3_va80/summaries/summary_train.md` | Train Reflexion KB construction summary for the VA80 next-trial KB. | Source for current ScienceWorld gated runs. |

## SQL

| Artifact | What it contains | Notes |
| --- | --- | --- |
| `experiment_reports/sql_transfer_radius/sql_transfer_radius_diagnostic.md` | Main paper-ready diagnostic and representative regressions. | Treat as the canonical SQL diagnostic narrative. |
| `experiment_reports/sql_transfer_radius/metrics_summary.csv` | Final metrics for the SQL gate. | Backing table for the diagnostic. |
| `experiment_reports/sql_transfer_radius/delta_status_summary.csv` | Regression/improvement counts by variant. | Useful for paired status counts. |
| `experiment_reports/sql_transfer_radius/regressions_vs_react.csv` | Largest regressions vs ReAct. | Useful for failure-mode audits. |
| `experiment_reports/sql_transfer_radius/representative_examples.csv` | Representative examples with retrieved memories. | Use for qualitative analysis. |

## Methodology and older reports

| Artifact | Status |
| --- | --- |
| `experiment_reports/context_retrieval_paper_content.md` | Methodology writeup for context retrieval. |
| `experiment_reports/tool_retrieval_paper.md` and `experiment_reports/tool_retrieval_paper_content.md` | Methodology writeups for tool retrieval. |
| `experiment_reports/react_baseline_report.md` | Superseded early ALFWorld baseline diagnostic. |
| `experiment_reports/reflexion_initial_results.md` | Superseded early Reflexion diagnostic. |
| `experiment_reports/ORGANIZATION_SUMMARY.md` | Historical repo organization note. |

## Destination for polished final runs

Future final, presentation-ready runs should go under `final_runs/`, following the contract in `final_runs/README.md`. Existing exploratory and historical outputs remain in their current locations.
