# Current State: 2026-05-23

This file preserves the experimental state as of 2026-05-23. No existing result artifacts were deleted, moved, or rewritten.

## Trusted current results

These are the safest artifacts to cite or build from right now.

| Result family | Trusted artifact(s) | Why trusted |
| --- | --- | --- |
| ScienceWorld dev-30 VA80 next-trial gate | `scienceworld_runs/memory_retrieval_v2/gates/dev_30_va80_next_trial_kb/summaries/summary_dev.md` | Broadest current ScienceWorld gate: 30 dev tasks, matched `max_valid_actions=80`, train-only Reflexion KB, and `VALID_NEXT_TRIAL`-filtered memory bank. |
| ScienceWorld dev-10 VA80 next-trial paired diagnostic | `experiment_reports/scienceworld_dev_10_va80_next_trial_gate.md`, `experiment_reports/scienceworld_dev_10_va80_next_trial_paired_deltas.csv` | Smaller but clearly documented paired gate with matched valid-action exposure and explicit deltas vs ReAct. |
| SQL transfer-radius diagnostic | `experiment_reports/sql_transfer_radius/sql_transfer_radius_diagnostic.md` plus CSVs in `experiment_reports/sql_transfer_radius/` | The report explicitly identifies the gate root, final metrics, paired delta statuses, and representative failure modes. |
| Retrieval methodology writeups | `experiment_reports/context_retrieval_paper_content.md`, `experiment_reports/tool_retrieval_paper.md`, `experiment_reports/tool_retrieval_paper_content.md` | Method descriptions rather than claims about final empirical lift. Useful for paper text after checking they still match implementation. |

### Current ScienceWorld readout

- Dev-30 VA80 next-trial gate: ReAct 8/30, CR 10/30, TR 11/30, CR+TR 9/30, hard-neg CR+TR 8/30.
- Dev-10 VA80 next-trial paired gate: ReAct 1/10, CR 3/10, TR 1/10, CR+TR 3/10, hard-neg CR+TR 2/10.
- The most conservative current statement is that ScienceWorld memory effects are positive for CR/TR in these gates, but small-sample and variant-sensitive.

### Current SQL readout

- SQL gate `intercode_sql_runs/gates/gpt-5.4/sql_action_stable_50`: ReAct 43/50, CR 41/50, TR 42/50, CR+TR 39/50, hard-neg CR+TR 41/50.
- The trusted interpretation is diagnostic rather than celebratory: SQL memories often have narrow transfer radius and can introduce plausible but wrong query-policy changes.

## Exploratory results

These are preserved and useful, but should be described as exploratory unless rerun in the final-run structure.

| Result family | Artifact(s) | Caveat |
| --- | --- | --- |
| ScienceWorld uncapped dev-10 gate | `experiment_reports/scienceworld_dev_10_uncapped_gate.md` and `scienceworld_runs/memory_retrieval_v2/gates/dev_10_uncapped_real_kb/` | Setup diagnostic only. KB had no `VALID_NEXT_TRIAL` entries and valid-action exposure was uncapped. |
| ALFWorld multiseed slices | `alfworld_runs/multiseed/gpt-5-mini/`, `alfworld_runs/multiseed/gemini-2.5-flash/` | Preserved summaries currently show single-framework slices, not a complete comparable variant matrix. |
| InterCode SQL multiseed slices | `intercode_sql_runs/multiseed/gpt-5-mini/`, `intercode_sql_runs/multiseed/gemini-2.5-flash/` | Useful run history, but current summary files are single-framework slices. Use the SQL transfer-radius gate for trusted SQL claims. |
| WebShop memory retrieval runs | `webshop_runs/memory_retrieval_v2/memory_agent_runs/` | Preserved dev/test hard-neg CR+TR summary. Needs final-run rerun for polished claims. |
| GPT-5.4 preflights and canaries | `alfworld_runs/preflight_gpt54_*`, `intercode_sql_runs/preflight_gpt54_*`, `intercode_sql_runs/multiseed_canary/gpt-5.4-mini/`, `webshop_runs/preflight_gpt54_*` | Treat as runner/model-preflight evidence, not final experiment results. |

## Superseded or historical results

These remain useful for debugging and provenance, but should not be cited as current empirical results.

| Artifact(s) | Why superseded |
| --- | --- |
| `experiment_reports/react_baseline_report.md` | Early ALFWorld/Gemini baseline diagnostic predating later runner and prompt work. |
| `experiment_reports/reflexion_initial_results.md` | Early Reflexion check; mechanism verification rather than current result. |
| `experiment_reports/ORGANIZATION_SUMMARY.md` | Historical organization note. |
| `alfworld_runs/memory_retrieval_v2/archive/` and `alfworld_runs/memory_retrieval_v2/misc/old_runs/` | Explicitly archived older ALFWorld outputs. Keep for provenance. |
| `scienceworld_runs/memory_retrieval_v2/smoke_reflexion/` and `scienceworld_runs/memory_retrieval_v2/kb_train_reflexion/` | Earlier smoke or one-trial KB construction; superseded for current ScienceWorld gates by `kb_train_reflexion_trials3_va80`. |

## Preservation notes

- Existing artifacts were left in place.
- The two previously untracked ScienceWorld report artifacts are now intended to be tracked:
  - `experiment_reports/scienceworld_dev_10_va80_next_trial_gate.md`
  - `experiment_reports/scienceworld_dev_10_va80_next_trial_paired_deltas.csv`
- Polished reruns should be placed under `final_runs/` using `final_runs/README.md`.

## Recommended next final-run targets

1. ScienceWorld: rerun the VA80 next-trial setup under `final_runs/scienceworld/` with a locked manifest and paired summary.
2. SQL: copy or rerun the trusted `sql_action_stable_50` diagnostic under `final_runs/sql/` once final reporting format is fixed.
3. ALFWorld/WebShop: rerun complete comparable variant matrices before using them as final paper claims.
