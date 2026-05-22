# ScienceWorld dev-10 uncapped gate

Run date: 2026-05-22

## Knowledge base

- Source run: `scienceworld_runs/memory_retrieval_v2/kb_train_reflexion/train/react_reflexion/seed_0`
- Split: train only
- Training tasks: 10 distinct ScienceWorld task types, one Reflexion trial each
- Reflexion training result: 2 / 10 success, avg score 0.2360
- Extracted memory bank: 20 entries
- Validation levels: 16 `VALID_SAME_TRIAL`, 4 `CANDIDATE`

## Gate

- Gate root: `scienceworld_runs/memory_retrieval_v2/gates/dev_10_uncapped_real_kb`
- Split: dev
- Tasks: 10 distinct ScienceWorld task types, one dev variation each
- Model: `gemini-2.5-flash`
- Embeddings: Gemini
- Valid-action exposure: uncapped (`--max-valid-actions 0`)

| Framework | Success | Avg score | Context retrieval | Help calls |
|---|---:|---:|---:|---:|
| ReAct | 5 / 10 | 0.3550 | 0 tasks / 0 learnings | 0 |
| CR | 2 / 10 | -0.0920 | 10 tasks / 60 learnings | 0 |
| TR | 3 / 10 | 0.3480 | 0 tasks / 0 learnings | 9 |
| CR+TR | 3 / 10 | 0.1650 | 10 tasks / 60 learnings | 20 |
| hard-neg CR+TR | 3 / 10 | 0.0780 | 10 tasks / 50 learnings | 8 |

## Readout

CR+TR did not beat ReAct on this gate: 3 / 10 vs 5 / 10 success, and 0.1650 vs 0.3550 average score.

Hard-negative CR+TR degraded relative to CR+TR by average score, 0.0780 vs 0.1650, but did not further degrade success count: both were 3 / 10.

## Interpretation caveat

This gate should be treated as a setup diagnostic, not as a decisive ScienceWorld result. The train memory bank was produced from one Reflexion trial per task (`max_trials=1`), so it contains no `VALID_NEXT_TRIAL` entries and mostly reflects partial same-trial fixes from failed trajectories. The dev gate therefore retrieved `VALID_SAME_TRIAL` memories rather than cross-trial-validated learnings, which is weaker than the validation standard used for the main gated runs.

The next fair ScienceWorld test should first build a multi-trial Reflexion KB with enough `VALID_NEXT_TRIAL` learnings, then rerun the dev gate with `--min-valid-level VALID_NEXT_TRIAL` and matched valid-action exposure between KB construction and evaluation.
