# ScienceWorld Final KB Report

## Result

Promoted the largest audited train-only ScienceWorld Reflexion candidate into the Phase 2 final KB path.

- Final KB directory: `/Users/rishisim/Documents/research/ltm-final-scienceworld-kb/final_runs/kb/scienceworld_train_reflexion_trials3_va80`
- Final KB JSON: `/Users/rishisim/Documents/research/ltm-final-scienceworld-kb/final_runs/kb/scienceworld_train_reflexion_trials3_va80/knowledge_base.json`
- Manifest: `/Users/rishisim/Documents/research/ltm-final-scienceworld-kb/final_runs/manifests/scienceworld_train_reflexion_trials3_va80.json`
- Audit summary: `/Users/rishisim/Documents/research/ltm-final-scienceworld-kb/final_runs/kb/scienceworld_train_reflexion_trials3_va80/audit_summary.json`
- Audit result: `PASS_WITH_COVERAGE_LIMITATION`
- Eval-use status: safe as a train-only 10-category KB, not safe as a complete all-30-category headline KB without rerun or explicit caveat.

## Source

- Source run: `/Users/rishisim/Documents/research/ltm-research/scienceworld_runs/memory_retrieval_v2/kb_train_reflexion_trials3_va80/train/react_reflexion/seed_0`
- Source suite config: `/Users/rishisim/Documents/research/ltm-research/scienceworld_runs/memory_retrieval_v2/kb_train_reflexion_trials3_va80/suite_config.json`
- Source original checkout commit: `91da9006684cb044ca8d4e67ed35383e20ef96ac`
- Phase 1 base commit in this worktree: `5d1fcf2e2a4c8d841ee101ccea44f1c1d6853eac`

Source command reconstructed from `suite_config.json`:

```bash
python3 scripts/run_scienceworld_suite.py --splits train --task-ids boil,change-the-state-of-matter-of,chemistry-mix,chemistry-mix-paint-secondary-color,chemistry-mix-paint-tertiary-color,find-animal,find-living-thing,find-non-living-thing,find-plant,freeze --num-tasks 10 --frameworks react_reflexion --max-variations-per-task 1 --model gemini-2.5-flash --embedding-provider gemini --runs-root scienceworld_runs/memory_retrieval_v2/kb_train_reflexion_trials3_va80 --max-valid-actions 80 --max-trials 3 --seed 0
```

Promotion/audit command was the one-off Python audit/promote script run from this worktree on 2026-05-23.

## Required Provenance Fields

- Model: `gemini-2.5-flash`
- Embedding provider: `gemini`
- Embedding model in copied cache: `gemini-embedding-001`
- Seed: `0`
- Split: `train`
- Framework list: `react_reflexion`
- Framework used for KB: `react_reflexion`
- Max trials: `3`
- Max valid actions: `80`
- Max variations per task: `1`
- KB path: `/Users/rishisim/Documents/research/ltm-final-scienceworld-kb/final_runs/kb/scienceworld_train_reflexion_trials3_va80/knowledge_base.json`
- max_learnings: `N/A_KB_GENERATION`
- min_valid_level: `N/A_KB_GENERATION`

## Coverage

- Intended ScienceWorld task categories: 30
- Covered task categories: 10
- Covered train variations: 10
- Covered categories: boil, change-the-state-of-matter-of, chemistry-mix, chemistry-mix-paint-secondary-color, chemistry-mix-paint-tertiary-color, find-animal, find-living-thing, find-non-living-thing, find-plant, freeze
- Missing categories: grow-fruit, grow-plant, identify-life-stages-1, identify-life-stages-2, inclined-plane-determine-angle, inclined-plane-friction-named-surfaces, inclined-plane-friction-unnamed-surfaces, lifespan-longest-lived, lifespan-longest-lived-then-shortest-lived, lifespan-shortest-lived, measure-melting-point-known-substance, measure-melting-point-unknown-substance, melt, mendelian-genetics-known-plant, mendelian-genetics-unknown-plant, power-component, power-component-renewable-vs-nonrenewable-energy, test-conductivity, test-conductivity-of-unknown-substances, use-thermometer

The source candidate therefore does not satisfy the preferred all-30-category ScienceWorld KB requirement. A full rebuild was not feasible in this turn because the active Python environment cannot import `scienceworld`, and `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`, and `OPENAI_API_KEY` are unset in the shell environment.

## Raw Audit

- Attempts: 10
- Trajectories: 20
- Reflexions: 20
- Progress task IDs: 10
- KB memories: 23
- VALID_NEXT_TRIAL memories: 7

Valid-level counts:

- CANDIDATE: 3
- VALID_NEXT_TRIAL: 7
- VALID_SAME_TRIAL: 13

Checks from raw artifacts:

- All trajectories have `split=train`: True
- Attempt task IDs match trajectory task IDs: True
- Progress matches trajectory task IDs: True
- KB references are a subset of train trajectory task IDs: True
- Required KB fields missing: 0
- Missing/corrupt JSON files: 0
- Success flag mismatches versus `reward >= 1.0`: 0
- KB reference step-range errors: 0
- Reward range observed across attempts/trajectories: [-1.00, 1.00]
- Issue embedding cache valid for copied KB hash/model/field: True
- Memory learning-count cache hash valid for copied KB: True
- Malformed extraction retries: not recorded by the existing preprocessing script/artifacts.

Recomputed from raw attempts:

- Success: 5 / 10
- Accuracy: 0.5000
- Average reward: 0.5980

## Caveats

This artifact preserves and manifests the cleanest available train-only ScienceWorld KB candidate, but it is coverage-limited. It should not be used for publication-grade all-30 ScienceWorld headline claims unless the orchestrator explicitly accepts the limitation or provisions the environment for a full all-30 train rebuild.
