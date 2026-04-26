# InterCode SQL Experimental Pipeline

## Overview

This directory holds all artifacts for the InterCode SQL (Spider) experiments
in the ACL 2026 SRW paper.  The pipeline has three phases that **must be run
in order**:

```
Phase 1 (train/react)       → raw trajectories for every train task
Phase 2 (train/reflexion)   → multi-trial trajectories + per-task reflexions
Phase 3 (KB extraction)     → knowledge_base.json from train trajectories only
Phase 4 (dev + test, all frameworks) → evaluation results
```

---

## Directory layout

```
intercode_sql_runs/
  memory_agent_runs/
    train/
      react/               # Phase 1 output
      react_reflexion/     # Phase 2 output
        knowledge_base.json   # Phase 3 output (built here)
        knowledge_base.csv
    dev/
      react/
      react_reflexion/
      react_cr/
      react_tr/
      react_cr_tr/
      react_hard_neg_cr_tr/
    test/
      (same structure as dev/)
    summaries/
      summary_dev.csv / .json / .md
      summary_test.csv / .json / .md
      summary_all.csv / .json
    suite_config.json
```

---

## Step-by-step instructions

### Prerequisites

1. Docker Desktop must be running.
2. Start the SQL container:
   ```bash
   docker-compose -f data/intercode_sql/docker/docker-compose.yml up -d
   ```
   Or just start the existing container:
   ```bash
   docker start docker-env-sql-spider_ic_ctr
   ```
3. Data splits must exist under `data/intercode_sql/`.  If not, run:
   ```bash
   python scripts/setup_intercode_sql.py
   ```

### Phase 1 — Train ReAct

Run ReAct on every train task to collect trajectories:

```bash
python scripts/run_intercode_sql_suite.py \
  --splits train \
  --frameworks react \
  --num-tasks 516 \
  --model gemini-2.5-flash \
  --runs-root intercode_sql_runs/memory_agent_runs
```

### Phase 2 — Train Reflexion

Run Reflexion on train tasks to produce multi-trial trajectories and reflexion
texts used by KB extraction:

```bash
python scripts/run_intercode_sql_suite.py \
  --splits train \
  --frameworks react_reflexion \
  --num-tasks 516 \
  --max-trials 3 \
  --model gemini-2.5-flash \
  --runs-root intercode_sql_runs/memory_agent_runs
```

### Phase 3 — KB extraction

Extract issue-learning pairs from **train** trajectories only.  The leakage
guard in `script.py` will raise an error if any `split=dev` or `split=test`
trajectories are detected.

```bash
PYTHONPATH=. python src/frameworks/memory_retrieval_v2/preprocessing/knowledge_base_v2/script.py \
  --log_dir intercode_sql_runs/memory_agent_runs/train/react_reflexion \
  --env intercode_sql \
  --resume
```

The KB is written to:
`intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json`

### Phase 4 — Dev / Test evaluation (all frameworks)

```bash
python scripts/run_intercode_sql_suite.py \
  --splits dev,test \
  --frameworks react,react_reflexion,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr \
  --num-tasks 200 \
  --model gemini-2.5-flash \
  --memory-bank intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json \
  --runs-root intercode_sql_runs/memory_agent_runs
```

Summaries land in `intercode_sql_runs/memory_agent_runs/summaries/`.

---

## Leakage guard

Every trajectory entry now carries a `"split"` field (`"train"`, `"dev"`, or
`"test"`).  `script.py` refuses to ingest any trajectory with
`split in ("dev", "test")` unless `--allow-eval-trajectories` is explicitly
passed.  This prevents accidental contamination of the KB with evaluation data.

---

## Smoke tests (verification only, not for scaling)

Quick functional checks — do not use these for paper results:

```bash
# 5-task ReAct smoke
python scripts/run_intercode_sql_suite.py \
  --splits dev --frameworks react --num-tasks 5 \
  --runs-root intercode_sql_runs/smoke_test

# 3-task CR+TR smoke (requires KB)
python scripts/run_intercode_sql_suite.py \
  --splits dev --frameworks react_cr_tr --num-tasks 3 \
  --memory-bank intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json \
  --runs-root intercode_sql_runs/smoke_test
```
