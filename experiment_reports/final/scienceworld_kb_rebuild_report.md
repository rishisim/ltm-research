# ScienceWorld KB Rebuild Report

## Status

`ABORTED_DO_NOT_USE`

The ScienceWorld Phase 2.5 trials3 rebuild was stopped immediately after the
protocol changed. No final 30-category ScienceWorld KB was produced or
promoted.

## Branch and Worktree

- Branch: `final/scienceworld-kb-rebuild`
- Worktree: `/Users/rishisim/Documents/research/ltm-scienceworld-kb-rebuild`
- Base commit before this report: `861aa83ece7b1e530f19d82c5571b6730e43addd`

## Preflight Completed Before Stop

- Created an isolated runtime at `/tmp/ltm-scienceworld-kb-venv`.
- Installed `scienceworld==1.2.3`, Gemini SDK packages, `python-dotenv`,
  `numpy`, and `openai==0.27.0` into that temporary venv.
- Verified `scienceworld` import and JVM-backed environment boot.
- Loaded API keys from sibling repo `.env` via `python-dotenv` without copying
  secrets into this worktree.
- Verified a 1-task ScienceWorld smoke run completed.
- Ran prepare/count checks:
  - 30/30 ScienceWorld categories available.
  - 30 tasks available for minimum 1 train variation/category.
  - 90 tasks available for preferred 3 train variations/category.

## Aborted Run

The stopped command was:

```bash
PYTHONPATH=. /tmp/ltm-scienceworld-kb-venv/bin/python scripts/run_scienceworld_suite.py \
  --splits train \
  --task-ids boil,change-the-state-of-matter-of,chemistry-mix,chemistry-mix-paint-secondary-color,chemistry-mix-paint-tertiary-color,find-animal,find-living-thing,find-non-living-thing,find-plant,freeze,grow-fruit,grow-plant,identify-life-stages-1,identify-life-stages-2,inclined-plane-determine-angle,inclined-plane-friction-named-surfaces,inclined-plane-friction-unnamed-surfaces,lifespan-longest-lived,lifespan-longest-lived-then-shortest-lived,lifespan-shortest-lived,measure-melting-point-known-substance,measure-melting-point-unknown-substance,melt,mendelian-genetics-known-plant,mendelian-genetics-unknown-plant,power-component,power-component-renewable-vs-nonrenewable-energy,test-conductivity,test-conductivity-of-unknown-substances,use-thermometer \
  --num-tasks 30 \
  --frameworks react_reflexion \
  --max-variations-per-task 1 \
  --model gemini-2.5-flash \
  --embedding-provider gemini \
  --runs-root final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat \
  --max-valid-actions 80 \
  --env-step-limit 100 \
  --max-trials 3 \
  --seed 0 \
  --quiet
```

The process was no longer running when checked after the interruption.

## Partial Artifacts Preserved

Partial artifacts are preserved here only as aborted/legacy provenance:

- `final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat/suite_config.json`
- `final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat/train/react_reflexion/seed_0/world.log`
- `final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat/train/react_reflexion/seed_0/reflexions.jsonl`
- `final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat/train/react_reflexion/seed_0/reflexions.json`
- `final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat/ABORTED_DO_NOT_USE.md`
- `final_runs/manifests/scienceworld_train_reflexion_trials3_va80_30cat_aborted.json`

Observed progress from `world.log`:

- Trial 1 started with 30 pending train tasks.
- Last completed line: `Trial 1 task #17: scienceworld_lifespan-longest-lived_var_0 - SUCCESS (score=1.00, steps=9)`.
- `reflexions.json` contains 18 entries for 18 unique task IDs, all from trial 1.

Missing final artifacts:

- `trajectories.json`
- `attempts.json`
- `metrics.json`
- `knowledge_base.json`
- embedding caches
- audit summary

## Eval Safety

This path is not safe for Phase 3. It contains no final KB and no complete
trajectory/attempt corpus. Do not use it for evaluation, aggregation, tables,
figures, or paper claims.

## Recommendation

Do not continue this naive/full trials3 ScienceWorld rebuild from this agent.
Wait for the orchestrator to assign the updated trials7 protocol in a fresh
instruction.
