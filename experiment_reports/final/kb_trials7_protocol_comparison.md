# Trials7 KB Protocol Comparison

Status: Phase 2.5 trials7 KBs promoted with caveats. Phase 3 eval has not started.

Integration branch: `final/integration-kb-state`

Integration worktree: `/Users/rishisim/Documents/research/ltm-research`

Merged source branches:

| Environment | Branch | Commit |
| --- | --- | --- |
| ALFWorld | `final/alfworld-kb-trials7` | `8964d5fe626bb6ed3042ef0de26fa2af8814ffc2` |
| SQL | `final/sql-kb-trials7` | `077252e2ecb6e6ae71ab8b3fdf0473cca8254a38` |
| ScienceWorld | `final/scienceworld-kb-trials7` | `99c31b803caaf10bf1bae431ab6f5e19bd785eae` |

## Common Protocol

Offline KB generation used train split only, `react_reflexion`, `max_trials=7`, `model=gemini-2.5-flash`, and Gemini embeddings. `max_learnings` and `min_valid_level` are eval-time retrieval settings, not KB generation filters.

Planned online eval remains held-out only, single-trial `react`, `react_cr`, `react_tr`, `react_cr_tr`, and `react_hard_neg_cr_tr`, with `min_valid_level=VALID_NEXT_TRIAL`, `max_learnings=5`, Gemini embeddings, and seeds `0,1,2`.

## Final KBs

| Environment | Safe for Phase 3 | Final KB | Coverage | Memories | VALID_NEXT_TRIAL |
| --- | --- | --- | --- | ---: | ---: |
| ALFWorld | yes | `final_runs/kb/alfworld_train_reflexion_trials7/knowledge_base.json` | 115/120 observed train games | 373 | 101 |
| SQL | yes | `final_runs/kb/sql_train_reflexion_trials7_sanitized/knowledge_base.sql_sanitized.json` | 516/516 train tasks | 415 | 92 |
| ScienceWorld | yes | `final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat/knowledge_base.json` | 30/30 categories, 1 variation/category | 129 | 34 |

## Audit Summary

| Environment | Train-only audit | Max trial | Duplicate trial keys | Quarantine / repair |
| --- | --- | ---: | ---: | --- |
| ALFWorld | Official ALFWorld cache split audit; valid_seen overlap 0, valid_unseen overlap 0 | 7 | not reported as duplicates; promoted refs all train | 31 malformed/ambiguous rows quarantined |
| SQL | Train/dev/test manifest audit; dev overlap 0, test overlap 0 | 7 | 0 duplicate `(task_id, trial_num)` pairs | 30 total memories quarantined, protocol contamination after sanitizer 0 |
| ScienceWorld | All trajectory rows tagged train; eval/test leakage splits empty | 7 | 0 duplicate `(task_id, variation, trial_num)` pairs | 5 malformed extraction groups repaired; unresolved malformed groups 0 |

## Environment Notes

ALFWorld was promoted from legacy trials7 artifacts instead of rebuilt. The original fixed 120-game manifest is absent, so this is clean train-only evidence for the 115 observed games rather than complete proof of the intended 120-game sample. The legacy seed is not recorded.

SQL used the intact trials3 train Reflexion run as a copied seed, then resumed the copy with `--max-trials 7`. The original trials3 run was not mutated. The copied trials7 run added 520 attempts, solved 25 additional train tasks, and ended with 396/516 train tasks solved.

ScienceWorld patched and validated safe Reflexion resume before building. The promoted artifact is the minimum acceptable 30-category KB with one train variation per category; prepare/count confirmed the preferred 90-task, 3-variation setting is available but was not run to bound remediation cost/runtime.

## Safe Next Step

Stop here until explicit user approval. The next safe action is Phase 3 final eval using only these trials7 KB paths and the held-out eval protocol above.
