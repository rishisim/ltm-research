# SQL Trials7 KB Report

## Verdict

SAFE FOR SQL PHASE 3 EVAL: yes, with the extraction caveats below.

The trials3 SQL train Reflexion run was copied into a separate trials7 build directory, resumed with `--resume --max-trials 7`, freshly extracted into a trials7 KB, sanitized, repaired for invalid refs, and promoted to `final_runs/kb/sql_train_reflexion_trials7_sanitized/`.

## Final Artifact

- Final KB: `final_runs/kb/sql_train_reflexion_trials7_sanitized/knowledge_base.sql_sanitized.json`
- Build dir: `final_runs/build/sql_train_reflexion_trials7/`
- Run dir: `final_runs/build/sql_train_reflexion_trials7/train/react_reflexion/seed_0/`
- Manifest: `final_runs/manifests/sql_train_reflexion_trials7_sanitized.json`
- Audit summary: `final_runs/kb/sql_train_reflexion_trials7_sanitized/audit_summary.json`
- Branch/worktree: `final/sql-kb-trials7` / `/Users/rishisim/Documents/research/ltm-sql-kb-trials7`
- Source-code commit used for generation: `9f9e395b6e6f4334757d9db7dd0a382075232688`

## Protocol

- Split: `train`
- Framework: `react_reflexion`
- Framework list: `['react_reflexion']`
- Model: `gemini-2.5-flash`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Seed: `0`
- Max trials: `7`
- Eval-time retrieval filter/cap: `VALID_NEXT_TRIAL`, fixed `max_learnings` (not applied during KB generation)

## Resume Evidence

- Source trials3 coverage: 516/516 train tasks; max trial 3; duplicate `(task_id, trial_num)` pairs: 0.
- Dry resume at `--max-trials 3`: reported 851 existing `(task, trial)` pairs and 371 already-succeeded tasks; no new attempts were added.
- Trials7 extension final counts: {'1': 516, '2': 179, '3': 156, '4': 145, '5': 130, '6': 124, '7': 121}.
- New attempts added: 520; tasks newly solved in trials 4-7: 25; final solved tasks: 396/516.
- Duplicate `(task_id, trial_num)` pairs after extension: 0.

## Train-Only Audit

- Canonical train/dev/test task counts: 516 / 200 / 200.
- Run unique task IDs: 516; missing train IDs: 0; extra nontrain IDs: 0.
- Held-out overlap: dev=0, test=0.
- Copied trials7 trajectories are split-tagged as: `['train']`.
- Success flag vs reward>=1.0 mismatches: 0.
- Reward range: [-1.0, 1.0].

## KB/Sanitizer Audit

- Raw trials7 memories: 445.
- Protocol-sanitized memories before ref repair: 416.
- Final sanitized memories promoted: 415.
- Protocol-contamination quarantine: 29.
- Invalid-ref repair quarantine: 1 (`unique_id=421`, impossible evidence trial 8).
- Final sanitized valid levels: {'VALID_SAME_TRIAL': 299, 'VALID_NEXT_TRIAL': 92, 'CANDIDATE': 24}.
- `VALID_NEXT_TRIAL` memories: 92.
- Sanitizer reclassification contamination count after promotion: 0.
- Sanitized issue/evidence dev/test overlap: issue dev=0, issue test=0, evidence dev=0, evidence test=0.
- Max issue/evidence trial in final sanitized KB: issue=7, evidence=7.
- Retrieval caches: learning-count tasks=199, issue embeddings=415, model=`gemini-embedding-001`.

## Caveats

- Fresh KB extraction had malformed JSON failures for `sql_547` and `sql_704` after retries; both are counted and recorded in the manifest.
- One post-sanitizer row had an impossible `evidence_ref.trial_num=8` and was quarantined before final promotion.
- Original trials3 artifacts were not modified; stale KB artifacts were removed only inside the copied trials7 build directory before fresh extraction.
- The original trials3 trajectories did not include split tags. The copied trials7 trajectories were stamped `split=train`, and train-only provenance is verified by exact task ID membership against the canonical manifests.

## Commands

See the manifest for full command strings. The key commands were:

```bash
/Users/rishisim/Documents/research/ltm-research/.venv/bin/dotenv -f /Users/rishisim/Documents/research/ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_intercode_sql_suite.py --splits train --frameworks react_reflexion --num-tasks 516 --max-trials 7 --model gemini-2.5-flash --embedding-provider gemini --runs-root final_runs/build/sql_train_reflexion_trials7 --seed 0 --resume --quiet
/Users/rishisim/Documents/research/ltm-research/.venv/bin/dotenv -f /Users/rishisim/Documents/research/ltm-research/.env run -- env PYTHONPATH=. LTM_EMBEDDING_PROVIDER=gemini /Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m src.frameworks.memory_retrieval_v2.preprocessing.knowledge_base_v2.script --log_dir final_runs/build/sql_train_reflexion_trials7/train/react_reflexion/seed_0 --env intercode_sql
/Users/rishisim/Documents/research/ltm-research/.venv/bin/dotenv -f /Users/rishisim/Documents/research/ltm-research/.env run -- env PYTHONPATH=. LTM_EMBEDDING_PROVIDER=gemini /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/utils/sql_memory_sanitizer.py --input final_runs/build/sql_train_reflexion_trials7/train/react_reflexion/seed_0/knowledge_base.json --output final_runs/build/sql_train_reflexion_trials7/train/react_reflexion/seed_0/knowledge_base.sql_sanitized.json
python helper to quarantine sanitized entries with issue/evidence trial_num outside [0, 7] into knowledge_base.sql_sanitized.ref_repair_quarantine.json
/Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/analysis/audit_intercode_sql_memory_contamination.py --memory-bank final_runs/kb/sql_train_reflexion_trials7_sanitized/knowledge_base.sql_sanitized.json
/Users/rishisim/Documents/research/ltm-research/.venv/bin/python -m py_compile scripts/run_intercode_sql_suite.py scripts/utils/sql_memory_sanitizer.py src/frameworks/memory_retrieval_v2/preprocessing/knowledge_base_v2/script.py
```
