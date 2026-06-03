# Phase 4.1 ScienceWorld Density Gate

## Status

`GATE_A_PROMISING_AMBIGUOUS`

The 3-train-variation ScienceWorld KB was built in the isolated Phase 4.1 path and evaluated on the requested held-out Gate A subset. Gate A does not improve success count, but `react_cr_tr` improves average reward over same-run ReAct and hard-neg remains below `react_cr_tr` on average reward. This is enough to treat the density hypothesis as plausible but not proven.

## Branch and Paths

- Branch: `final/scienceworld-density-gate`
- Worktree: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate`
- Report commit at generation: `076fc7ac143925cffd0faa3356d7bffd4b34c427`
- Candidate KB: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate/final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var/knowledge_base.json`
- Candidate eval: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate/final_runs/phase4_1/eval/scienceworld_trials7_gemini_3var_gate`
- Old 1-var comparison source: `/Users/rishisim/Documents/research/ltm-scienceworld-density-gate/final_runs/eval/scienceworld_trials7_gemini`

## Commands

```text
prepare/count: /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m dotenv -f ../ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_scienceworld_suite.py --splits train --task-ids <30 categories> --num-tasks 90 --max-variations-per-task 3 --frameworks react_reflexion --max-trials 7 --max-valid-actions 80 --model gemini-2.5-flash --embedding-provider gemini --runs-root final_runs/phase4_1/prepare_count/scienceworld_train_3var_count --seed 0 --prepare-only --resume --quiet
copy/resume seed: cp -a final_runs/kb/scienceworld_train_reflexion_trials7_va80_30cat final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var; remove generated KB/cache/audit artifacts only inside the candidate copy
KB resume fill: /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m dotenv -f ../ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_scienceworld_suite.py --splits train --task-ids <30 categories> --num-tasks 90 --max-variations-per-task 3 --frameworks react_reflexion --max-trials 7 --max-valid-actions 80 --model gemini-2.5-flash --embedding-provider gemini --runs-root final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var --seed 0 --resume --quiet
initial extraction: /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m dotenv -f ../ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m src.frameworks.memory_retrieval_v2.preprocessing.knowledge_base_v2.script --log_dir final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var/train/react_reflexion/seed_0 --env scienceworld --resume
targeted extraction repair: Removed scienceworld_find-animal_var_2 from candidate knowledge_base_progress.json, then reran extractor with --task_ids scienceworld_find-animal_var_2 --resume; retry succeeded and added 7 entries.
cache build: LTM_EMBEDDING_PROVIDER=gemini /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m dotenv -f ../ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python - <<'PY' create_knowledge_base_embeddings(kb, embed_field='issue_text', force=True); build_learning_counts_table(kb, force=True) PY
Gate A eval: /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m dotenv -f ../ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_scienceworld_suite.py --splits test --task-ids <first 15 categories> --num-tasks 15 --max-variations-per-task 1 --frameworks react,react_cr_tr,react_hard_neg_cr_tr --max-valid-actions 80 --max-learnings 5 --min-valid-level VALID_NEXT_TRIAL --model gemini-2.5-flash --embedding-provider gemini --memory-bank final_runs/phase4_1/kb/scienceworld_train_reflexion_trials7_va80_30cat_3var/knowledge_base.json --runs-root final_runs/phase4_1/eval/scienceworld_trials7_gemini_3var_gate --seed 0 --resume --quiet
```

## Prepare Count

- Selected train tasks: 90 / 90
- Categories: 30 / 30
- Each category selected variations `[0, 1, 2]`: True

## Candidate KB Audit

| Check | Value | Note |
|---|---|---|
| Train selected tasks | 90 | 90 expected |
| Categories | 30 | 30 expected |
| Trajectory rows | 401 | deduped raw trajectories |
| Duplicate task/variation/trial keys | 0 | 0 expected |
| Max trial | 7 | 7 expected |
| Split counts | {'train': 401} | train only |
| KB entries | 394 | after repair |
| VALID_NEXT_TRIAL entries | 137 | retrieval threshold eligible |
| Malformed rows | 0 | 0 expected |
| Reference errors | 0 | 0 expected |

Valid-level counts: `{'VALID_SAME_TRIAL': 223, 'VALID_NEXT_TRIAL': 137, 'CANDIDATE': 34}`

Cache validity:

- Issue embeddings: 394 entries, `gemini-embedding-001`, source hash matches: True
- Learning counts: 60 unique task descriptions, 394 learnings, `gemini-embedding-001`, source hash matches: True

Extraction repair:

- Initial extraction: 2 JSON parse retries, unresolved group `scienceworld_find-animal_var_2`.
- Targeted repair: removed only that failed id from candidate progress, reran targeted extraction with `--resume`, 1 retry, 7 entries added.
- Unresolved failures after repair: none.

## Gate A Metrics

Metrics below are recomputed from raw `attempts.json` files, not summaries. The old 1-var rows are filtered to the identical 15 task ids used by the 3-var gate.

| Framework | Source | Success | Avg Reward | Avg Steps |
|---|---|---|---|---|
| react | old 1-var Stage C | 3 / 15 | 0.3173 | 38.40 |
| react | new 3-var Gate A | 3 / 15 | 0.0153 | 34.20 |
| react_cr_tr | old 1-var Stage C | 3 / 15 | 0.1933 | 36.67 |
| react_cr_tr | new 3-var Gate A | 3 / 15 | 0.2227 | 33.13 |
| react_hard_neg_cr_tr | old 1-var Stage C | 3 / 15 | -0.0480 | 32.27 |
| react_hard_neg_cr_tr | new 3-var Gate A | 3 / 15 | 0.1547 | 37.07 |

Matched task ids:

```text
scienceworld_boil_var_21
scienceworld_change-the-state-of-matter-of_var_21
scienceworld_chemistry-mix_var_24
scienceworld_chemistry-mix-paint-secondary-color_var_27
scienceworld_chemistry-mix-paint-tertiary-color_var_27
scienceworld_find-animal_var_225
scienceworld_find-living-thing_var_225
scienceworld_find-non-living-thing_var_225
scienceworld_find-plant_var_225
scienceworld_freeze_var_21
scienceworld_grow-fruit_var_93
scienceworld_grow-plant_var_93
scienceworld_identify-life-stages-1_var_9
scienceworld_identify-life-stages-2_var_6
scienceworld_inclined-plane-determine-angle_var_126
```

## Decision

- `react_cr_tr` beats same-run ReAct on success: False
- `react_cr_tr` beats same-run ReAct on average reward: True
- hard-neg below `react_cr_tr` on success: False
- hard-neg below `react_cr_tr` on average reward: True

Interpretation: Supports underbuilt-KB as a plausible contributor only weakly/ambiguously: denser KB improved react_cr_tr average reward versus both same-run ReAct and old 1-var react_cr_tr on the matched subset, but did not improve success count.

## Phase 5 Recommendation

Recommendation: keep ScienceWorld as a diagnostic with caveat for the Phase 5 rewrite unless a follow-up Gate B confirms the signal. Because Gate A is promising/ambiguous rather than a clear fail, the next approved experiment should be Gate B: all 30 categories, seeds 0/1/2, all five frameworks.
