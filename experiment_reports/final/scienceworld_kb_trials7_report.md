# ScienceWorld Trials7 KB Report

## Status

`PASS_PROMOTED_MINIMUM_VARIATION_COVERAGE`

Promoted a ScienceWorld train-only Reflexion KB for the trials7 final protocol.

- Final KB directory: `/Users/rishisim/Documents/research/ltm-scienceworld-kb-trials7/final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat`
- Final KB JSON: `/Users/rishisim/Documents/research/ltm-scienceworld-kb-trials7/final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat/knowledge_base.json`
- Raw run directory: `/Users/rishisim/Documents/research/ltm-scienceworld-kb-trials7/final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat/train/react_reflexion/seed_0`
- Manifest: `/Users/rishisim/Documents/research/ltm-scienceworld-kb-trials7/final_runs/manifests/scienceworld_train_reflexion_trials7_va80_30cat.json`
- Audit summary: `/Users/rishisim/Documents/research/ltm-scienceworld-kb-trials7/final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat/audit_summary.json`

## Protocol

- Split: train only
- Framework: react_reflexion
- Model: gemini-2.5-flash
- Embedding provider: gemini
- Seed: 0
- Max trials: 7
- Max valid actions: 80
- Max variations per task: 1
- Max learnings / min valid level: N/A for offline KB generation

## Resume Patch Validation

- Synthetic validation reconstructed prior trajectories/reflexions, skipped completed `(task_id, variation, trial_num)` pairs, preserved a solved task, appended only the missing trials for the pending task, and ended with zero duplicate keys.
- Real copied-smoke validation resumed a 1-task run from max_trials=1 to max_trials=2, skipped trial 1, appended trial 2 only, and ended with keys `[('scienceworld_boil_var_0', 0, 1), ('scienceworld_boil_var_0', 0, 2)]` and zero duplicates.

## Coverage and Raw Run Audit

- Categories covered: 30 / 30
- Train variations covered: 30
- Prepare/count result: 90 train tasks available for 3/category; this run used the minimum accepted 1/category setting.
- Trajectories: 125
- Attempts: 30
- Reflexion records: 125
- Max trial observed: 7
- Duplicate `(task_id, variation, trial_num)` pairs: 0
- Split counts: {'train': 125}
- Eval/test leakage splits: []
- Final train Reflexion success: 16 / 30 (avg reward 0.4700)

## KB Audit

- Total memories: 129
- VALID_NEXT_TRIAL memories: 34
- Valid-level counts: {'VALID_SAME_TRIAL': 80, 'VALID_NEXT_TRIAL': 34, 'CANDIDATE': 15}
- KB referenced task IDs: 30 / 30
- Malformed KB rows after repair: 0
- Reference errors after repair: 0
- Issue embedding cache valid: True (gemini-embedding-001)
- Learning-count cache valid: True (gemini-embedding-001)

## Extraction Repair

Initial extraction produced 66 entries and had 5 task groups fail JSON parsing after 11 retry events. After increasing ScienceWorld extraction output budget to 8192 tokens and retrying only those task groups, all five succeeded and the final KB contains 129 entries.

## Caveats

- This is the minimum acceptable 30-category headline KB: 1 train variation per category, not the preferred 3/category KB.
- The final pushed commit containing this report should be used as the artifact commit; the generation-start commit was `9f9e395b6e6f4334757d9db7dd0a382075232688`.
