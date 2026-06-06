# Phase 4.1 ScienceWorld Density Gate

## Status

`GATE_B_COMPLETED_DIAGNOSTIC_ONLY`

Gate B completed for all 30 ScienceWorld test categories, seeds 0/1/2, and all five frameworks using the Phase 4.1 3-var KB. The result does **not** promote ScienceWorld to a headline final result: `react_cr_tr` is below same-run ReAct on mean success and mean reward, and hard-neg beats or matches `react_cr_tr` on success in every seed.

## Branch and Paths

- Branch: `final/scienceworld-density-gate`
- Worktree: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate`
- Report commit at generation: `5509e00a0b61a0501466cc4fa9a096b5dd2fa22b`
- Candidate KB: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate/final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var/knowledge_base.json`
- Candidate eval root: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate/final_runs/phase4_1/eval/scienceworld_trials7_gemini_3var_gate`
- Old 1-var Stage C source: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate/final_runs/eval/scienceworld_trials7_gemini`

## Commands

```text
Seed 0: /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m dotenv -f ../ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_scienceworld_suite.py --splits test --task-ids <30 categories> --num-tasks 30 --max-variations-per-task 1 --frameworks react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr --max-valid-actions 80 --max-learnings 5 --min-valid-level VALID_NEXT_TRIAL --model gemini-2.5-flash --embedding-provider gemini --memory-bank final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var/knowledge_base.json --runs-root final_runs/phase4_1/eval/scienceworld_trials7_gemini_3var_gate --seed 0 --resume --quiet
Seed 1: same command with --seed 1
Seed 2: same command with --seed 2
Audit: Recomputed metrics, deltas, retrieval counts, and old/new comparisons from raw attempts and retrieval JSON files.
```

## Config Audit

- Suite config points to 3-var KB: True
- Split/test, 30 tasks, all five frameworks: True
- Retrieval settings: max_learnings=5, min_valid_level=`VALID_NEXT_TRIAL`
- Model/embedding: `gemini-2.5-flash`, `gemini`
- Audit errors: `[]`

## Metrics By Seed

Raw attempts are the source of truth.

| Seed | Framework | Success | Accuracy | Avg Reward | Avg Steps |
|---|---|---|---|---|---|
| 0 | react | 11 / 30 | 0.3667 | 0.2787 | 31.70 |
| 0 | react_cr | 7 / 30 | 0.2333 | 0.1043 | 28.53 |
| 0 | react_tr | 11 / 30 | 0.3667 | 0.4193 | 33.83 |
| 0 | react_cr_tr | 8 / 30 | 0.2667 | 0.1750 | 31.63 |
| 0 | react_hard_neg_cr_tr | 11 / 30 | 0.3667 | 0.3163 | 32.70 |
| 1 | react | 10 / 30 | 0.3333 | 0.2900 | 30.90 |
| 1 | react_cr | 9 / 30 | 0.3000 | 0.2407 | 32.30 |
| 1 | react_tr | 10 / 30 | 0.3333 | 0.3287 | 33.40 |
| 1 | react_cr_tr | 8 / 30 | 0.2667 | 0.2737 | 32.13 |
| 1 | react_hard_neg_cr_tr | 11 / 30 | 0.3667 | 0.2230 | 33.77 |
| 2 | react | 9 / 30 | 0.3000 | 0.3207 | 33.90 |
| 2 | react_cr | 8 / 30 | 0.2667 | 0.2740 | 29.87 |
| 2 | react_tr | 8 / 30 | 0.2667 | 0.1770 | 34.13 |
| 2 | react_cr_tr | 11 / 30 | 0.3667 | 0.3520 | 29.30 |
| 2 | react_hard_neg_cr_tr | 11 / 30 | 0.3667 | 0.3753 | 35.93 |

## Aggregate Mean/Std

| Framework | Success Mean | Success Std | Accuracy Mean | Accuracy Std | Reward Mean | Reward Std |
|---|---|---|---|---|---|---|
| react | 10.00 | 1.00 | 0.3333 | 0.0333 | 0.2964 | 0.0217 |
| react_cr | 8.00 | 1.00 | 0.2667 | 0.0333 | 0.2063 | 0.0899 |
| react_tr | 9.67 | 1.53 | 0.3222 | 0.0509 | 0.3083 | 0.1224 |
| react_cr_tr | 9.00 | 1.73 | 0.3000 | 0.0577 | 0.2669 | 0.0887 |
| react_hard_neg_cr_tr | 11.00 | 0.00 | 0.3667 | 0.0000 | 0.3049 | 0.0768 |

## Paired Deltas Vs ReAct

Deltas are paired by seed and computed from raw attempts.

| Framework | Success Deltas s0/s1/s2 | Mean Success Delta | Reward Deltas s0/s1/s2 | Mean Reward Delta |
|---|---|---|---|---|
| react_cr | -4, -1, -1 | -2.00 | -0.1743, -0.0493, -0.0467 | -0.0901 |
| react_tr | 0, 0, -1 | -0.33 | 0.1407, 0.0387, -0.1437 | 0.0119 |
| react_cr_tr | -3, -2, 2 | -1.00 | -0.1037, -0.0163, 0.0313 | -0.0296 |
| react_hard_neg_cr_tr | 0, 1, 2 | 1.00 | 0.0377, -0.0670, 0.0547 | 0.0084 |

## Old 1-var Vs New 3-var

Both sides use the same 30 categories and seeds 0/1/2. Metrics are recomputed from raw attempts.

| Framework | Old Success Mean | Old Reward Mean | New Success Mean | New Reward Mean | Success Delta | Reward Delta |
|---|---|---|---|---|---|---|
| react | 9.00 | 0.2142 | 10.00 | 0.2964 | 1.00 | 0.0822 |
| react_cr | 8.00 | 0.1710 | 8.00 | 0.2063 | 0.00 | 0.0353 |
| react_tr | 8.00 | 0.2560 | 9.67 | 0.3083 | 1.67 | 0.0523 |
| react_cr_tr | 8.33 | 0.2122 | 9.00 | 0.2669 | 0.67 | 0.0547 |
| react_hard_neg_cr_tr | 8.67 | 0.2598 | 11.00 | 0.3049 | 2.33 | 0.0451 |

## Retrieval Audit

Context-memory variants retrieved memories for every task. All retrieved memories used `VALID_NEXT_TRIAL` rows. TR-only wrote 30 agent trajectory rows per seed but did not make help/tool calls, so no context retrieval file is expected for `react_tr`.

| Seed | Framework | JSON Records | JSONL Records | Min Pool | Max Pool | Selected Count Dist | Valid Levels |
|---|---|---|---|---|---|---|---|
| 0 | react_cr | 30 | 30 | 1 | 39 | {5: 18, 2: 5, 1: 1, 4: 4, 3: 2} | {'VALID_NEXT_TRIAL': 323} |
| 1 | react_cr | 30 | 30 | 1 | 39 | {5: 18, 2: 5, 1: 1, 4: 4, 3: 2} | {'VALID_NEXT_TRIAL': 323} |
| 2 | react_cr | 30 | 30 | 1 | 39 | {5: 18, 2: 5, 1: 1, 4: 4, 3: 2} | {'VALID_NEXT_TRIAL': 323} |
| 0 | react_cr_tr | 30 | 30 | 1 | 39 | {5: 18, 2: 5, 1: 1, 4: 4, 3: 2} | {'VALID_NEXT_TRIAL': 323} |
| 1 | react_cr_tr | 30 | 30 | 1 | 39 | {5: 18, 2: 5, 1: 1, 4: 4, 3: 2} | {'VALID_NEXT_TRIAL': 323} |
| 2 | react_cr_tr | 30 | 30 | 1 | 39 | {5: 18, 2: 5, 1: 1, 4: 4, 3: 2} | {'VALID_NEXT_TRIAL': 323} |
| 0 | react_hard_neg_cr_tr | 30 | 30 | 3 | 38 | {5: 29, 3: 1} | {'VALID_NEXT_TRIAL': 307} |
| 1 | react_hard_neg_cr_tr | 30 | 30 | 3 | 38 | {5: 29, 3: 1} | {'VALID_NEXT_TRIAL': 307} |
| 2 | react_hard_neg_cr_tr | 30 | 30 | 3 | 38 | {5: 29, 3: 1} | {'VALID_NEXT_TRIAL': 307} |

## Decision

- `react_cr_tr` beats ReAct on mean success: False
- `react_cr_tr` beats ReAct on mean reward: False
- hard-neg below `react_cr_tr` on mean success: False
- hard-neg below `react_cr_tr` on mean reward: False
- hard-neg beats or matches `react_cr_tr` on success: True

Stability:

- `react_cr_tr` success deltas vs ReAct by seed: [-3, -2, 2]
- `react_cr_tr` reward deltas vs ReAct by seed: [-0.1037, -0.0163, 0.0313]
- hard-neg minus `react_cr_tr` success by seed: [3, 3, 0]

Interpretation: Gate B does not promote ScienceWorld: react_cr_tr is below ReAct on mean success and mean reward, and hard-neg beats or matches react_cr_tr on success in every seed. The 3-var KB slightly improves react_cr_tr over the old 1-var KB, but not enough to clear the same-run promotion gate.

## Phase 5 Recommendation

Keep ScienceWorld as diagnostic evidence with a density caveat; do not make it a headline final result unless a later method changes the hard-neg/CR+TR ordering.
