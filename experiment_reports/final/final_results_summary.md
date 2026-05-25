# Final Results Summary

## Provenance

- Branch/worktree: `final/tables-figures` at `/Users/rishisim/Documents/research/ltm-tables-figures`
- Git commit at generation time: `202f4f3b5fac8c446682a1d52c496ebf88c654aa`
- Aggregation command: `scripts/analysis/aggregate_final_runs.py --runs-root final_runs/eval --output-dir experiment_reports/final/aggregates`
- Table/figure command: `scripts/analysis/generate_final_tables_figures.py`
- Inputs: `final_runs/eval/*_trials7_gemini`, `final_runs/kb/*trials7*`, `final_runs/audits/phase3_stage_C_audit.json`
- Outputs: `paper/tables/final_*.tex`, `paper/figures/final_*.pdf`, `experiment_reports/final/aggregates/*`

## Aggregate Counts

- Suite roots: 3
- Per-seed rows: 45
- Framework aggregate rows: 15
- Paired-delta rows: 36
- Stage C audit: PASS with 0 errors and 0 warnings.

## Headline Stage C Results

| Environment | Best framework | Best success | ReAct success | Best delta |
| --- | --- | ---: | ---: | ---: |
| ALFWorld | ReAct + CR + TR | 82.8% | 67.9% | +14.9 pp |
| SQL | ReAct + CR + TR | 92.0% | 91.5% | +0.5 pp |
| ScienceWorld | ReAct | 30.0% | 30.0% | +0.0 pp |

## Generated Tables

- `paper/tables/final_main_results.tex`
- `paper/tables/final_kb_inventory.tex`
- `paper/tables/final_memory_usage.tex`
- `paper/tables/final_paired_deltas.tex`

## Generated Figures

- `paper/figures/final_gain_over_react.pdf`
- `paper/figures/final_reward_vs_cost.pdf`
- `paper/figures/final_paired_delta_heatmap.pdf`
- `paper/figures/final_hard_negative_effect.pdf`

## Caveats

- ALFWorld trials7 KB is train-only after quarantine, but the legacy exact 120-game train manifest is absent; audit observes 115 unique train trajectory task IDs and no valid_seen/valid_unseen overlap.
- ScienceWorld final KB covers all 30 categories with the minimum accepted 1 train variation per category, not the preferred 3 variations per category.
- Provider dollar cost is not available in final raw logs; `final_reward_vs_cost.pdf` uses average steps per task as a trajectory-cost proxy.
- SQL `react_cr` task `sql_288` had recovered bare code-fence actions in seeds 0, 1, and 2; the environment rejected those actions and the task succeeded, so this is documented as non-fatal model-formatting noise.
- Diagnostic ScienceWorld repair runs under `diagnostics/` are preserved but ignored by audit and aggregation discovery.
- No retrieval-budget ablation was present under `final_runs/eval`, so the optional retrieval-budget plot was not generated.
