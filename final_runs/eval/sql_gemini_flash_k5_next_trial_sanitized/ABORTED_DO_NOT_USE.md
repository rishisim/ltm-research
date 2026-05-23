# Aborted SQL Eval: Do Not Use

Status: aborted / superseded.

This directory contains partial artifacts from an interrupted SQL evaluation that
used the trials3 sanitized KB. The protocol changed to require trials7 SQL KB
generation before final SQL eval, so these artifacts are preserved only as
legacy provenance and must not be promoted, aggregated, or cited as final
results.

- Branch: `final/sql-eval`
- Worktree: `/Users/rishisim/Documents/research/ltm-sql-final-eval`
- Base commit at launch: `861aa83ece7b1e530f19d82c5571b6730e43addd`
- Abort recorded at: `2026-05-23T21:41:44Z`
- Attempted suite: `seed_0_suite`
- Completed finalized metrics: none
- Partial task log: ReAct seed 0 wrote 96 task lines before interruption.

Superseded command:

```bash
/Users/rishisim/Documents/research/ltm-research/.venv/bin/dotenv -f /Users/rishisim/Documents/research/ltm-research/.env run -- /Users/rishisim/Documents/research/ltm-research/.venv/bin/python scripts/run_intercode_sql_suite.py --num-tasks 200 --splits test --frameworks react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr --model gemini-2.5-flash --embedding-provider gemini --memory-bank final_runs/kb/sql_train_reflexion_trials3_sanitized/knowledge_base.sql_sanitized.json --runs-root final_runs/eval/sql_gemini_flash_k5_next_trial_sanitized/seed_0_suite --reward-threshold 1.0 --max-learnings 5 --min-valid-level VALID_NEXT_TRIAL --seed 0 --quiet
```
