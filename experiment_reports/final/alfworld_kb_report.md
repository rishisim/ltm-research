# ALFWorld KB Audit Report

Status: blocked, not promoted.

Worktree: `/Users/rishisim/Documents/research/ltm-final-alfworld-kb`

Branch: `final/alfworld-kb`

Audit base commit: `5d1fcf2e2a4c8d841ee101ccea44f1c1d6853eac`

## Source Artifacts Audited

- Source KB: `alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json`
- Source CSV: `alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.csv`
- Embedding/count caches:
  - `alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.mem_learning_counts.json`
  - `alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.issue_embeddings_cache.json`
- Raw provenance:
  - `alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/trajectories.json`
  - `alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/reflexions.json`
  - `alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/knowledge_base_progress.json`

## Required Final Provenance

| Field | Required / final campaign value | Source audit finding |
|---|---:|---|
| model | `gemini-2.5-flash` | Inferred from `scripts/legacy/run_memory_allocation.py`; not recorded in KB JSON |
| embedding provider | `gemini` | Supported by cache metadata `gemini-embedding-001` |
| seed | required | Not recorded |
| split | `train` | Verified by task IDs against official ALFWorld cache, but split tags are missing from trajectories |
| framework | `react_reflexion` | Inferred from `MemoryAllocationReflexion`; not recorded in raw rows |
| framework list | `react_reflexion` | Inferred |
| max_trials | `3` | Not satisfied; observed raw trajectory/reflexion trials through trial 7 |
| max_learnings | N/A for KB construction | Eval-only field |
| min_valid_level | N/A for KB construction | Eval-only field; final eval should use `VALID_NEXT_TRIAL` |
| final KB path | `final_runs/kb/alfworld_train_reflexion_trials3/` | Created only with `BLOCKED_DO_NOT_USE.md`; no KB promoted |

## Audit Evidence

The source JSON files are parseable, and the source KB contains 404 memories with no duplicate `unique_id` values.

Memory counts by `valid_level`:

| valid_level | count |
|---|---:|
| `VALID_SAME_TRIAL` | 259 |
| `VALID_NEXT_TRIAL` | 103 |
| `CANDIDATE` | 41 |
| blank | 1 |

Raw trajectory/reflexion provenance:

| Item | Count |
|---|---:|
| trajectory rows | 251 |
| trajectory unique task IDs | 115 |
| reflexion rows | 251 |
| reflexion unique task IDs | 115 |
| progress entries | 155 |
| progress entries with full `task/trial` IDs | 115 |
| progress entries with base task IDs only | 40 |

Trajectory rows by trial:

| trial | rows |
|---:|---:|
| 1 | 115 |
| 2 | 49 |
| 3 | 28 |
| 4 | 23 |
| 5 | 15 |
| 6 | 12 |
| 7 | 9 |

Task coverage by unique trajectory task type:

| task type | unique task IDs |
|---|---:|
| `look_at_obj_in_light` | 20 |
| `pick_and_place_simple` | 19 |
| `pick_clean_then_place_in_recep` | 20 |
| `pick_cool_then_place_in_recep` | 19 |
| `pick_heat_then_place_in_recep` | 17 |
| `pick_two_obj_and_place` | 20 |

Split audit:

- Against `/Users/rishisim/.cache/alfworld/json_2.1.1`, all 115 trajectory task IDs map to official `train`.
- Overlap with official `valid_seen`: 0.
- Overlap with official `valid_unseen`: 0.
- The trajectory rows themselves contain no `split` field.
- The current untracked `alfworld_mini` sample at `/Users/rishisim/Documents/research/ltm-research/alfworld_mini` differs from the source KB sample; only 5 source trajectory task IDs overlap its current train sample. This means the original 120-game sampled manifest cannot be reconstructed from the current worktree state.

Trial-depth audit:

- Max observed trajectory trial: 7.
- Max observed KB reference trial: 7.
- KB entries with any issue/evidence reference after trial 3: 85.
- `VALID_NEXT_TRIAL` entries with max reference trial <= 3: 56.
- `VALID_NEXT_TRIAL` entries with max reference trial > 3: 47.

Malformed-row audit:

- Blank `valid_level`: 1 row, `unique_id=182`.
- Empty `learning_text`: 2 rows.
- `learning_text == "No specific learning found."`: 1 row.
- Empty `task_desc`: 1 row.
- Empty evidence task IDs: 24 rows.
- Empty issue task IDs: 1 row.
- One candidate row uses evidence `task_id=N/A`, `trial_num=-1`; one candidate row uses evidence `trial_num=0`.

## Decision

The existing ALFWorld KB is not clean enough to promote as `final_runs/kb/alfworld_train_reflexion_trials3/knowledge_base.json`.

Positive finding: the source appears train-only with respect to the official ALFWorld split map, with zero official valid_seen/valid_unseen overlap in trajectory task IDs.

Blocking findings:

1. It covers 115 unique trajectory task IDs, not the intended 120 train games.
2. It is not a `max_trials=3` KB. Source trajectories/reflexions extend through trial 7, and 85 KB rows cite references after trial 3.
3. Split tags, seed, exact source generation command, and the stable sampled 120-game task manifest are not recorded in the source raw rows.
4. The KB contains malformed or empty rows, including one blank `valid_level`.

I did not copy `knowledge_base.json` into `final_runs/kb/alfworld_train_reflexion_trials3/`. That path contains only a blocker note and is not safe for eval use.

## Rebuild Feasibility

A clean rebuild should be run before Phase 3 using:

- split: `train`
- framework: `react_reflexion`
- model: `gemini-2.5-flash`
- embedding provider: `gemini`
- max_trials: `3`
- intended task count: 120 train games, from a preserved fixed manifest

I did not start the rebuild in this subagent because it requires a fresh LLM campaign of up to 120 x 3 ALFWorld trials. The gitignored `alfworld_mini` dataset is absent from this worktree, and the current untracked sample in the original workspace differs from the source KB sample. Rebuild should begin only after the orchestrator fixes or approves the exact train manifest and cost/runtime budget.

## Commands Run

```bash
pwd && git status --short --branch && git rev-parse HEAD && git branch --show-current
rg --files alfworld_runs/memory_retrieval_v2/knowledge_base final_runs experiment_reports scripts/analysis | sed -n '1,200p'
find alfworld_runs/memory_retrieval_v2/knowledge_base -maxdepth 3 -type f | sort | sed -n '1,240p'
wc -c alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/trajectories.json alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/reflexions.json alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/knowledge_base_progress.json alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.csv alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.mem_learning_counts.json alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.issue_embeddings_cache.json
sed -n '1,5p' alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.csv
rg -n "alfworld|train|valid_seen|valid_unseen|120|split|memory_retrieval_v2|knowledge_base" -S scripts configs . | sed -n '1,240p'
find . -maxdepth 4 -type f \( -name '*alfworld*' -o -name '*memory*' -o -name '*retrieval*' -o -name '*reflexion*' \) | sort | sed -n '1,300p'
sed -n '1,260p' scripts/legacy/run_memory_allocation.py
sed -n '260,560p' scripts/legacy/run_memory_allocation.py
python3 - <<'PY'
# Loaded and summarized source KB, trajectories, reflexions, progress, cache metadata, split overlaps, trial references, malformed rows, and sha256 hashes.
PY
date -u +%Y-%m-%dT%H:%M:%SZ && mkdir -p final_runs/kb/alfworld_train_reflexion_trials3 final_runs/manifests experiment_reports/final
python3 -m py_compile scripts/legacy/run_memory_allocation.py scripts/analysis/audit_final_run.py scripts/analysis/aggregate_final_runs.py
python3 -m json.tool alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json >/dev/null
python3 -m json.tool alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/trajectories.json >/dev/null
python3 -m json.tool alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/reflexions.json >/dev/null
python3 -m json.tool alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/knowledge_base_progress.json >/dev/null
python3 -m json.tool final_runs/manifests/alfworld_train_reflexion_trials3.json >/dev/null
```

## Source Hashes

| file | sha256 |
|---|---|
| `knowledge_base.json` | `bf7e8b96ba7aa76f63b0b9ace6b5a9ccb44b8bd8536d05980ace9b3fa14d0ed8` |
| `knowledge_base.csv` | `de8745ac9b1ab91dc0cfec3d6381f813ae6b38a8c8a523647b1d5d1b9ce13395` |
| `knowledge_base.mem_learning_counts.json` | `08578f12e3a4e24d7fd4b9f16ec7df20d339eab3145efabb0390c58c842c481b` |
| `knowledge_base.issue_embeddings_cache.json` | `e9772787390675c11e9410a17026c3e7c5bb4499e0986ff87deb44f6d0ac0d2d` |
| `trajectories.json` | `04fd3f0702553c35c7706238a0acc9bf3bdd65de61961492db1bfaf323c573d2` |
| `reflexions.json` | `37a5af85c51c962a20d017e07979ebac7b862bf66bcfa506e6c14ed1daeaa6d8` |
| `knowledge_base_progress.json` | `e65cac775b46526bff92fec93205a7a61eebf68d96e728a2069ec0cbbf11a38f` |
