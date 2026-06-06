# TextWorld Phase 4.2 Environment Gate

Date: 2026-06-03

## Verdict

FAIL. Do not advance this TextWorld configuration to a larger paper-claim gate.
Proceed to Phase 5 with the existing environment story unless a materially
harder TextWorld-style method is pre-registered as a new gate.

## Scope

This was an exploratory environment-selection gate, not a final paper result.
Artifacts are confined to:

- `textworld_runs/phase4_2_textworld_gate/`

No `final_runs/`, Phase 4 table, or paper figure artifact was modified.

## Pre-Registered Protocol

- TextWorld generated `.ulx` games only.
- Train seeds: 42000-42011.
- Eval seeds: 52000-52011.
- Generation parameters: `nb_rooms=4`, `nb_objects=8`, `quest_length=3`, `max_steps=50`.
- Train-only KB from `react_reflexion`, `max_trials=3`.
- Held-out eval frameworks: `react`, `react_cr_tr`, `react_hard_neg_cr_tr`.
- Model: `gemini-2.5-flash`.
- Embedding provider: `gemini`.
- No paper claim unless all gate checks pass.

## Results

| Framework | Success / Total | Accuracy | Avg Reward | Avg Steps/Task |
|---|---:|---:|---:|---:|
| ReAct | 10 / 12 | 0.8333 | 0.8333 | 10.7500 |
| ReAct + CR + TR | 11 / 12 | 0.9167 | 0.9167 | 11.0833 |
| ReAct + hard-neg CR + TR | 10 / 12 | 0.8333 | 0.8333 | 13.2500 |

Train Reflexion solved 12 / 12 by trial 3 and produced 14 KB entries.

## Gate Checks

Passed:

- Train/eval task overlap: zero.
- Train/eval seed overlap: zero.
- Hidden walkthrough leakage: none detected.
- KB refs train-only: yes.
- Eval trajectory completeness: 100%.

Failed:

- ReAct not floor/ceiling: failed, ReAct was 10 / 12 (83.3%).
- CR+TR beats ReAct by at least two held-out tasks: failed, delta was +1.
- CR+TR beats hard-neg by at least two held-out tasks: failed, delta was +1.

## Interpretation

The integration is feasible, but this TextWorld setting is too easy for the
current model and does not expose a strong relevance gap. Hard-neg remains close
to CR+TR, so the setting does not add a clean positive-transfer environment under
the predeclared criteria.

Recommendation: do not spend Phase 5 time expanding this run. If TextWorld is
revisited, pre-register a harder generator profile first, such as longer quests,
more rooms, distractor-heavy object placement, or a TextWorld-style custom split
that requires reusable mechanics without making the baseline saturate.

## Key Artifacts

- Pre-registration: `textworld_runs/phase4_2_textworld_gate/PRE_REGISTRATION.md`
- Gate audit: `textworld_runs/phase4_2_textworld_gate/gate_audit.json`
- Summary: `textworld_runs/phase4_2_textworld_gate/summaries/summary_eval.md`
- Train KB: `textworld_runs/phase4_2_textworld_gate/train/react_reflexion/seed_0/knowledge_base.json`
- Runner: `scripts/run_textworld_phase42_gate.py`
