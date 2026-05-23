# ScienceWorld dev-10 VALID_NEXT_TRIAL gate

Run date: 2026-05-22

## Knowledge base

- Source run: `scienceworld_runs/memory_retrieval_v2/kb_train_reflexion_trials3_va80/train/react_reflexion/seed_0`
- Split: train only
- Training tasks: 10 distinct ScienceWorld task types
- Reflexion trials: max 3
- Valid-action exposure: `--max-valid-actions 80`
- Train Reflexion result: 5 / 10 success, avg score 0.5980
- Extracted memory bank: 23 entries
- Validation levels: 13 `VALID_SAME_TRIAL`, 7 `VALID_NEXT_TRIAL`, 3 `CANDIDATE`
- Note: `scienceworld_boil_var_0` extraction repeatedly returned malformed JSON and contributed no entries.

## Gate

- Gate root: `scienceworld_runs/memory_retrieval_v2/gates/dev_10_va80_next_trial_kb`
- Split: dev
- Tasks: 10 distinct ScienceWorld task types
- Model: `gemini-2.5-flash`
- Embeddings: Gemini
- Valid-action exposure: `--max-valid-actions 80`
- Retrieval filter: `--min-valid-level VALID_NEXT_TRIAL`

| Framework | Success | Avg score | Avg steps | Context retrieval | Help calls |
|---|---:|---:|---:|---:|---:|
| ReAct | 1 / 10 | 0.1270 | 32.2 | 0 tasks / 0 learnings | 0 |
| CR | 3 / 10 | 0.5100 | 39.6 | 10 tasks / 41 learnings | 0 |
| TR | 1 / 10 | -0.1530 | 29.4 | 0 tasks / 0 learnings | 10 |
| CR+TR | 3 / 10 | 0.3050 | 38.1 | 10 tasks / 41 learnings | 10 |
| hard-neg CR+TR | 2 / 10 | 0.0530 | 31.0 | 10 tasks / 42 learnings | 20 |

## Paired deltas vs ReAct

| Framework | Success improvements | Success regressions | Same success status | Avg score delta |
|---|---:|---:|---:|---:|
| CR | 2 | 0 | 8 | +0.3830 |
| TR | 1 | 1 | 8 | -0.2800 |
| CR+TR | 2 | 0 | 8 | +0.1780 |
| hard-neg CR+TR | 1 | 0 | 9 | -0.0740 |

## Readout

CR+TR beats ReAct under the matched `max-valid-actions=80` setup: 3 / 10 vs 1 / 10 success and 0.3050 vs 0.1270 avg score.

Hard-negative CR+TR degrades relative to CR+TR: 2 / 10 vs 3 / 10 success and 0.0530 vs 0.3050 avg score.
