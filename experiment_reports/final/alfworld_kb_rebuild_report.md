# ALFWorld KB Rebuild Report

SAFE FOR FINAL EVAL: no.

The ALFWorld final train KB rebuild was not started. The prepare/count gate
found missing runtime dependencies, missing task data, and missing API keys, so
starting the 120-task LLM rebuild would have failed immediately and risked
leaving a partial final artifact.

## Branch and Worktree

- Branch: `final/alfworld-kb-rebuild`
- Worktree: `/Users/rishisim/Documents/research/ltm-alfworld-kb-rebuild`
- Base at prepare time: `861aa83ece7b1e530f19d82c5571b6730e43addd`

## Intended Final Configuration

- Artifact: `alfworld_train_reflexion_trials3`
- Final KB directory: `final_runs/kb/alfworld_train_reflexion_trials3/`
- Manifest: `final_runs/manifests/alfworld_train_reflexion_trials3.json`
- Split: `train`
- Required train coverage: 120 / 120 ALFWorld train games
- Framework: `react_reflexion`
- Model: `gemini-2.5-flash`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Seed: `0`
- `max_trials`: `3`
- `max_learnings`: `N/A_KB_construction_eval_only`
- `min_valid_level`: `N/A_KB_construction_eval_only`

## Prepare Command

```bash
python3 scripts/alfworld_final_kb_builder.py --mode prepare --final-dir final_runs/kb/alfworld_train_reflexion_trials3 --manifest final_runs/manifests/alfworld_train_reflexion_trials3.json --expected-tasks 120 --max-trials 3 --model gemini-2.5-flash --embedding-provider gemini --seed 0
```

Exit code: `2`, expected for a blocked prepare gate.

## Blockers

- Python module `alfworld` is not importable.
- No ALFWorld task data root was found. `ALFWORLD_DATA` is unset and this worktree does not contain `alfworld_mini/`, `data/json_2.1.1/train`, or another discoverable train split.
- `LTM_OPENROUTER_API_KEY` is missing for `gemini-2.5-flash` chat calls routed through OpenRouter.
- `GEMINI_API_KEY` is missing for `gemini-embedding-001` cache construction.

Observed environment:

- `ALFWORLD_DATA`: missing
- `LTM_OPENROUTER_API_KEY`: missing
- `GEMINI_API_KEY`: missing
- `GOOGLE_API_KEY`: missing
- `OPENAI_API_KEY`: missing
- `alfworld` import: failed
- `google.genai` import: available
- `google.generativeai` import: available

## Scope Estimate

The prepare gate estimates the intended rebuild as:

- 120 train tasks
- 3 trials/task
- 360 maximum agent episodes
- 17,640 maximum agent action LLM calls
- 360 maximum Reflexion LLM calls
- 120 KB extraction LLM calls
- 18,120 conservative upper-bound LLM calls

The runtime estimate is 10-25 hours before KB extraction at 2-5 seconds per
action call. This is a conservative upper bound because episodes stop after
success or environment termination.

## Artifacts Written

- `scripts/alfworld_final_kb_builder.py`
- `final_runs/kb/alfworld_train_reflexion_trials3/BLOCKED_REBUILD_DO_NOT_USE.md`
- `final_runs/kb/alfworld_train_reflexion_trials3/prepare_summary.json`
- `final_runs/manifests/alfworld_train_reflexion_trials3.json`
- `experiment_reports/final/alfworld_kb_rebuild_report.md`

The existing Phase 2 blocker marker remains preserved:

- `final_runs/kb/alfworld_train_reflexion_trials3/BLOCKED_DO_NOT_USE.md`

## Rebuild Command After Blockers Are Fixed

After installing/importing ALFWorld, making the exact 120-game train split
available, and exporting both `LTM_OPENROUTER_API_KEY` and `GEMINI_API_KEY`, run:

```bash
python3 scripts/alfworld_final_kb_builder.py --mode run --final-dir final_runs/kb/alfworld_train_reflexion_trials3 --manifest final_runs/manifests/alfworld_train_reflexion_trials3.json --expected-tasks 120 --max-trials 3 --model gemini-2.5-flash --embedding-provider gemini --seed 0
```

That command will regenerate trajectories/reflexions in
`final_runs/kb/alfworld_train_reflexion_trials3/raw_react_reflexion_run/`, extract
the KB, build Gemini embedding caches, and run the final audit.

## Validation Run

- `python3 -m py_compile scripts/alfworld_final_kb_builder.py`
- `python3 -m json.tool final_runs/kb/alfworld_train_reflexion_trials3/prepare_summary.json`
- `python3 -m json.tool final_runs/manifests/alfworld_train_reflexion_trials3.json`

## Recommendation

Do not use ALFWorld in Phase 3 yet. Provision the ALFWorld package/data and
required API keys, then rerun the rebuild command above. If cost approval is
needed, use the recorded 360-episode / 18,120-call upper-bound estimate before
starting the expensive run.
