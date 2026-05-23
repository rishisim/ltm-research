# SQL Final KB Report

## Verdict

SAFE FOR FINAL EVAL: yes, with the split-tag caveat below.

The final SQL KB is a preserved sanitized train-only InterCode SQL KB copied from the original checkout into `final_runs/kb/sql_train_reflexion_trials3_sanitized/`. The sanitized bank has 336 memories, no sanitizer-detected protocol contamination, and is backed by source trajectories covering all 516 canonical train tasks with zero overlap against dev/test task IDs.

## Final Artifact

- Final KB path: `final_runs/kb/sql_train_reflexion_trials3_sanitized/knowledge_base.sql_sanitized.json`
- Manifest: `final_runs/manifests/sql_train_reflexion_trials3_sanitized.json`
- Audit summary copy: `final_runs/kb/sql_train_reflexion_trials3_sanitized/audit_summary.json`
- Worktree: `/Users/rishisim/Documents/research/ltm-final-sql-kb`
- Branch: `final/sql-kb`
- Code commit at packaging: `5d1fcf2e2a4c8d841ee101ccea44f1c1d6853eac`
- Source checkout commit at audit: `91da9006684cb044ca8d4e67ed35383e20ef96ac`

## Required Provenance

- Source run directory: `/Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion`
- Source KB path: `/Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.sql_sanitized.json`
- Model: `gemini-2.5-flash`
- Embedding provider: `gemini`
- Embedding model evidence: `gemini-embedding-001`
- Seed: not recorded in the legacy source `suite_config.json`; source task order follows the canonical train manifest order.
- Split: `train`
- Framework: `react_reflexion`
- Framework list: `['react_reflexion']`
- Max trials: `3`
- max_learnings: `N/A` for KB artifact; Phase 3 eval should use `5`.
- min_valid_level: `N/A` for KB artifact; Phase 3 eval should use `VALID_NEXT_TRIAL`.

## Source Commands

Commands recorded or reconstructed from the source run README/config:

```bash
python scripts/run_intercode_sql_suite.py --splits train --frameworks react_reflexion --num-tasks 516 --max-trials 3 --model gemini-2.5-flash --runs-root intercode_sql_runs/memory_agent_runs
PYTHONPATH=. python src/frameworks/memory_retrieval_v2/preprocessing/knowledge_base_v2/script.py --log_dir intercode_sql_runs/memory_agent_runs/train/react_reflexion --env intercode_sql --resume
python3 scripts/utils/sql_memory_sanitizer.py --input /Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json --output /tmp/sql_kb_sanitizer_repro.json
```

## Train-Only Audit Evidence

- Canonical train manifest: 516 tasks across 8 databases.
- Canonical dev/test manifests: 200 dev tasks, 200 test tasks.
- Manifest overlap: train/dev=0, train/test=0, dev/test=0.
- Source trajectories: 851 rows, 516 unique task IDs, 516 unique task indices.
- Source reflexions: 851 rows, 516 unique task IDs.
- KB progress: 516 entries, 516 unique IDs.
- Missing train IDs from trajectories: 0.
- Extra trajectory IDs outside train: 0.
- Trajectory overlap with dev/test IDs: dev=0, test=0.
- Progress missing/extra vs train: missing=0, extra=0.
- Trajectory split tags present: 0 of 851 rows.

Caveat: the consolidated source `trajectories.json` rows do not include `split` fields. I therefore verified train-only status from the run path, `suite_config.json`, `knowledge_base_progress.json`, and exact membership of every trajectory/reflexion/progress `sql_<index>` in `data/intercode_sql/train.json`, with zero dev/test overlap.

## Trajectory/Metric Audit

- Trial counts: {1: 516, 2: 179, 3: 156}
- Max trial: 3
- Tasks with more than 3 attempts: 0
- Success rows: 371; rows with reward >= 1.0: 371
- Success flag vs reward>=1.0 mismatches: 0
- Reward range: [-1.0, 1.0]
- Source final train success from metrics: 371 / 516 accuracy=0.7190

## Sanitization/Contamination Audit

- Raw KB memories: 359
- Sanitized kept memories: 336
- Quarantined memories: 23
- Sanitized KB contamination audit: 0 quarantined memories when reclassified.
- Raw KB contamination audit reason counts: {'answer_after_submit_protocol': 3, 'data_insufficiency_answer_protocol': 2, 'final_answer_protocol': 7, 'non_sql_answer_protocol': 12, 'runner_evaluation_artifact': 2, 'submit_answer_protocol': 20, 'submit_colon_empty_protocol': 2}
- Sanitizer reproduction: sanitized KB and quarantine reproduced byte-for-byte in `/tmp`.
- Sanitized valid levels: {'CANDIDATE': 32, 'VALID_NEXT_TRIAL': 67, 'VALID_SAME_TRIAL': 237}
- Raw valid levels: {'CANDIDATE': 41, 'VALID_NEXT_TRIAL': 76, 'VALID_SAME_TRIAL': 242}
- Quarantine valid levels: {'CANDIDATE': 9, 'VALID_NEXT_TRIAL': 9, 'VALID_SAME_TRIAL': 5}
- Sanitized unique ID duplicates: 0
- Sanitized issue/evidence refs overlap dev/test: issue dev=0, issue test=0, evidence dev=0, evidence test=0.

## Copied Files

- `knowledge_base.sql_sanitized.json`
- `knowledge_base.sql_sanitized.report.json`
- `knowledge_base.sql_sanitized.quarantine.json`
- `knowledge_base.sql_sanitized.mem_learning_counts.json`
- `knowledge_base.sql_sanitized.issue_embeddings_cache.json`
- `audit_summary.json`

## Validation Commands Run

```bash
python3 scripts/analysis/audit_intercode_sql_memory_contamination.py --memory-bank /Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.sql_sanitized.json
python3 scripts/analysis/audit_intercode_sql_memory_contamination.py --memory-bank /Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json
python3 scripts/utils/sql_memory_sanitizer.py --input /Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json --output /tmp/sql_kb_sanitizer_repro.json
cmp -s /tmp/sql_kb_sanitizer_repro.json /Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.sql_sanitized.json
cmp -s /tmp/sql_kb_sanitizer_repro.quarantine.json /Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.sql_sanitized.quarantine.json
python3 -m py_compile scripts/utils/sql_memory_sanitizer.py scripts/analysis/audit_intercode_sql_memory_contamination.py
```

## Caveats

- The source consolidated trajectory rows lack split tags; see train-only audit evidence above.
- The legacy source suite config does not record a seed or embedding provider field. The chat model is recorded as `gemini-2.5-flash`; Gemini embedding provenance is recorded in copied cache files as `gemini-embedding-001`.
- This phase did not run final evals. Phase 3 should point SQL eval memory to `final_runs/kb/sql_train_reflexion_trials3_sanitized/knowledge_base.sql_sanitized.json` and use `max_learnings=5`, `min_valid_level=VALID_NEXT_TRIAL`, `embedding_provider=gemini`.
