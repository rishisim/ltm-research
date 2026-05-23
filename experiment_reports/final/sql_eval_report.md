# SQL Eval Report

## Status

ABORTED / DO NOT USE.

The SQL eval background lane was stopped on request because the protocol changed
to trials7 SQL KB generation before final SQL evaluation. The partial trials3-KB
eval artifacts are preserved for provenance only and are not final, not audited
as final, and not safe for aggregation or paper claims.

## Branch / Worktree

- Branch: `final/sql-eval`
- Worktree: `/Users/rishisim/Documents/research/ltm-sql-final-eval`
- Base commit at launch: `861aa83ece7b1e530f19d82c5571b6730e43addd`
- Abort recorded at: `2026-05-23T21:41:44Z`

## What Was Running

```bash
/Users/rishisim/Documents/research/ltm-research/.venv/bin/dotenv -f /Users/rishisim/Documents/research/ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_intercode_sql_suite.py --num-tasks 200 --splits test --frameworks react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr --model gemini-2.5-flash --embedding-provider gemini --memory-bank final_runs/kb/sql_train_reflexion_trials3_sanitized/knowledge_base.sql_sanitized.json --runs-root final_runs/eval/sql_gemini_flash_k5_next_trial_sanitized/seed_0_suite --reward-threshold 1.0 --max-learnings 5 --min-valid-level VALID_NEXT_TRIAL --seed 0 --quiet
```

The process was no longer present in the process table when checked after the
interrupt. No additional SQL eval commands were started.

## Partial Artifacts

- Abort marker: `final_runs/eval/sql_gemini_flash_k5_next_trial_sanitized/ABORTED_DO_NOT_USE.md`
- Manifest: `final_runs/manifests/sql_eval_aborted_trials3_superseded.json`
- Suite config: `final_runs/eval/sql_gemini_flash_k5_next_trial_sanitized/seed_0_suite/suite_config.json`
- Partial ReAct log: `final_runs/eval/sql_gemini_flash_k5_next_trial_sanitized/seed_0_suite/test/react/seed_0/world.log`
- Runner stdout/stderr log: `final_runs/eval/sql_gemini_flash_k5_next_trial_sanitized/logs/seed_0_runner.log`

Observed partial progress:

- ReAct seed 0 wrote 96 task log lines.
- No `metrics.json` files were completed.
- No `trajectories.json` or memory-agent JSONL trajectories were completed.
- Memory frameworks did not start.

## Preflight Already Completed Before Abort

- Docker Desktop and `docker-env-sql-spider` image were available.
- InterCode SQL DB reset smoke passed.
- OpenRouter Gemini chat smoke passed.
- Gemini embedding smoke passed with `gemini-embedding-001`.
- A one-task runner smoke in `/tmp/ltm_sql_final_eval_smoke` passed `audit_final_run.py` with 0 errors, but it was only preflight evidence and is not part of final results.

## Recommendation

Do not resume or aggregate this trials3-KB eval. Start a fresh SQL final eval
only after the trials7 SQL KB protocol is implemented and explicitly assigned.
