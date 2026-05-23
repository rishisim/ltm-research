# ALFWorld Trials7 KB Audit and Promotion Report

Status: promoted, safe_for_eval true.

Worktree: `/Users/rishisim/Documents/research/ltm-alfworld-kb-trials7`

Branch: `final/alfworld-kb-trials7`

Generation git commit: `70626b99e0376bf97afbe39aeb703c4ca56f4cdc`

## Final Artifact

- Final KB path: `final_runs/kb/alfworld_train_reflexion_trials7/knowledge_base.json`
- Manifest path: `final_runs/manifests/alfworld_train_reflexion_trials7.json`
- Quarantine path: `final_runs/kb/alfworld_train_reflexion_trials7/knowledge_base.quarantine.json`
- Audit summary path: `final_runs/kb/alfworld_train_reflexion_trials7/audit_summary.json`

## Protocol

| Field | Value |
| --- | --- |
| offline framework | react_reflexion |
| split | train |
| max_trials | 7 |
| model | gemini-2.5-flash |
| embedding_provider | gemini |
| embedding_model | gemini-embedding-001 |
| eval max_learnings | N/A_KB_generation_eval_fixed_later |
| eval min_valid_level | N/A_KB_generation_eval_uses_VALID_NEXT_TRIAL_later |
| seed | not_recorded_in_legacy_source |

## Train-Only Provenance

- Method: `official ALFWorld cache walk with base-task fallback for legacy refs`
- Official split root: `/Users/rishisim/.cache/alfworld/json_2.1.1`
- Official split counts: train=3553, valid_seen=140, valid_unseen=134
- Trajectory full task IDs mapped to train: 115/115
- Trajectory overlap with valid_seen: 0
- Trajectory overlap with valid_unseen: 0
- Promoted KB refs with non-train or unknown split: 0
- Limitation: Original sampled 120-game train manifest is absent; exact missing game IDs cannot be reconstructed from repo-local artifacts. Split membership is verified against the official ALFWorld cache.

## Coverage

- Intended train games: 120
- Observed unique trajectory task IDs: 115
- Coverage complete: False

| Task type | Observed unique task IDs | Intended |
| --- | --- | --- |
| look_at_obj_in_light | 20 | 20 |
| pick_and_place_simple | 19 | 20 |
| pick_clean_then_place_in_recep | 20 | 20 |
| pick_cool_then_place_in_recep | 19 | 20 |
| pick_heat_then_place_in_recep | 17 | 20 |
| pick_two_obj_and_place | 20 | 20 |

## Row Audit

- Source KB entries: 404
- Promoted clean entries: 373
- Quarantined entries: 31
- Max observed source trial: 7
- Max promoted reference trial: 7

Source valid levels:

| valid_level | count |
| --- | --- |
| VALID_SAME_TRIAL | 259 |
| VALID_NEXT_TRIAL | 103 |
| CANDIDATE | 41 |
| blank | 1 |

Promoted valid levels:

| valid_level | count |
| --- | --- |
| VALID_SAME_TRIAL | 256 |
| VALID_NEXT_TRIAL | 101 |
| CANDIDATE | 16 |

Quarantine reasons:

| reason | count |
| --- | --- |
| issue_ref_split_ambiguous | 6 |
| evidence_ref_split_ambiguous | 6 |
| evidence_ref_split_unknown | 1 |
| missing_evidence_ref_trial_num | 24 |
| missing_task_desc | 1 |
| non_informative_learning_text | 5 |
| invalid_valid_level | 1 |
| missing_issue_ref_task_id | 1 |
| missing_issue_ref_trial_num | 1 |
| missing_evidence_ref_task_id | 24 |
| evidence_ref_trial_out_of_range | 1 |

## Commands

- `python3 scripts/analysis/promote_alfworld_trials7_kb.py`

## Caveats

- The original fixed 120-game train manifest is not present in repo-local artifacts; legacy trajectories cover 115/120 intended train games.
- Observed coverage is uneven across task types; see coverage table.
