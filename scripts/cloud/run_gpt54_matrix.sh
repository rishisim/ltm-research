#!/usr/bin/env bash
set -euo pipefail

# Run the clean GPT-5.4 replacement matrix.
# Default order: WebShop first, then ALFWorld, then SQL.
# The runs are restartable; every suite call uses --resume.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.4}"
SEEDS="${SEEDS:-0 1 2}"
FRAMEWORKS="${FRAMEWORKS:-react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr}"
PHASES="${PHASES:-webshop,alfworld,sql}"
WEBSHOP_WORKERS="${WEBSHOP_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-logs/gpt54_rerun}"
WEBSHOP_NUM_TASKS="${WEBSHOP_NUM_TASKS:-${NUM_TASKS:-200}}"
ALF_NUM_TASKS="${ALF_NUM_TASKS:-134}"
SQL_NUM_TASKS="${SQL_NUM_TASKS:-${NUM_TASKS:-200}}"
WEBSHOP_SPLITS="${WEBSHOP_SPLITS:-test}"
SQL_SPLITS="${SQL_SPLITS:-test}"
WEBSHOP_MAX_LEARNINGS="${WEBSHOP_MAX_LEARNINGS:-3}"
WEBSHOP_MIN_VALID_LEVEL="${WEBSHOP_MIN_VALID_LEVEL:-VALID_NEXT_TRIAL}"
ALF_MAX_LEARNINGS="${ALF_MAX_LEARNINGS:-3}"
ALF_MIN_VALID_LEVEL="${ALF_MIN_VALID_LEVEL:-VALID_NEXT_TRIAL}"
SQL_MAX_LEARNINGS="${SQL_MAX_LEARNINGS:-3}"
SQL_MIN_VALID_LEVEL="${SQL_MIN_VALID_LEVEL:-VALID_NEXT_TRIAL}"

WEB_ROOT="${WEB_ROOT:-webshop_runs/rerun_clean/${MODEL}}"
ALF_ROOT="${ALF_ROOT:-alfworld_runs/rerun_clean/${MODEL}}"
SQL_ROOT="${SQL_ROOT:-intercode_sql_runs/rerun_clean/${MODEL}}"
WEB_KB_ORIGINAL="${WEB_KB_ORIGINAL:-webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train/knowledge_base.json}"
WEB_KB_SANITIZED="${WEB_KB_SANITIZED:-webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train/knowledge_base.sanitized.json}"
ALF_KB="${ALF_KB:-alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json}"
SQL_KB_ORIGINAL="${SQL_KB_ORIGINAL:-intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json}"
SQL_KB_SANITIZED="${SQL_KB_SANITIZED:-intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.sql_sanitized.json}"

mkdir -p "${LOG_DIR}"

phase_enabled() {
  [[ ",${PHASES}," == *",$1,"* ]]
}

run_logged() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo
  echo "============================================================"
  echo "[run] ${name}"
  echo "[run] log=${log_file}"
  echo "============================================================"
  "$@" 2>&1 | tee -a "${log_file}"
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "[run] required path missing: $1" >&2
    exit 1
  fi
}

require_file ".env"
if phase_enabled webshop; then
  require_file "${WEB_KB_ORIGINAL}"
  echo "[run] sanitizing WebShop memory bank"
  .venv/bin/python scripts/analysis/audit_webshop_kb.py \
    --memory-bank "${WEB_KB_ORIGINAL}" \
    --report-json "${WEB_KB_ORIGINAL%.json}.audit.json" \
    --report-csv "${WEB_KB_ORIGINAL%.json}.audit.csv" \
    --write-sanitized "${WEB_KB_SANITIZED}" \
    --exclude-product-specific
  require_file "${WEB_KB_SANITIZED}"
fi
if phase_enabled alfworld; then
  require_file "${ALF_KB}"
fi
if phase_enabled sql; then
  require_file "${SQL_KB_ORIGINAL}"
  echo "[run] sanitizing SQL memory bank"
  .venv/bin/python scripts/utils/sql_memory_sanitizer.py \
    --input "${SQL_KB_ORIGINAL}" \
    --output "${SQL_KB_SANITIZED}"
  require_file "${SQL_KB_SANITIZED}"
fi

echo "[run] model=${MODEL}"
echo "[run] seeds=${SEEDS}"
echo "[run] frameworks=${FRAMEWORKS}"
echo "[run] phases=${PHASES}"
echo "[run] webshop_workers=${WEBSHOP_WORKERS}"
echo "[run] webshop_num_tasks=${WEBSHOP_NUM_TASKS}"
echo "[run] alf_num_tasks=${ALF_NUM_TASKS}"
echo "[run] sql_num_tasks=${SQL_NUM_TASKS}"
echo "[run] webshop_retrieval=max${WEBSHOP_MAX_LEARNINGS},min=${WEBSHOP_MIN_VALID_LEVEL}"
echo "[run] alf_retrieval=max${ALF_MAX_LEARNINGS},min=${ALF_MIN_VALID_LEVEL}"
echo "[run] sql_retrieval=max${SQL_MAX_LEARNINGS},min=${SQL_MIN_VALID_LEVEL}"

if phase_enabled sql; then
  echo "[run] ensuring SQL Docker is up"
  docker compose -f data/intercode_sql/docker/docker-compose.yml up -d
fi

if phase_enabled webshop; then
  echo "[run] phase 1: WebShop"
  for seed in ${SEEDS}; do
    run_logged "webshop_seed_${seed}" \
      ./scripts/run_webshop_in_docker.sh \
        --num-tasks "${WEBSHOP_NUM_TASKS}" \
        --splits "${WEBSHOP_SPLITS}" \
        --frameworks "${FRAMEWORKS}" \
        --model "${MODEL}" \
        --seed "${seed}" \
        --runs-root "${WEB_ROOT}" \
        --memory-bank "${WEB_KB_ORIGINAL}" \
        --sanitized-memory-bank "${WEB_KB_SANITIZED}" \
        --max-learnings "${WEBSHOP_MAX_LEARNINGS}" \
        --min-valid-level "${WEBSHOP_MIN_VALID_LEVEL}" \
        --workers "${WEBSHOP_WORKERS}" \
        --quiet \
        --resume
  done
fi

if phase_enabled alfworld; then
  echo "[run] phase 2: ALFWorld"
  for seed in ${SEEDS}; do
    run_logged "alfworld_seed_${seed}" \
      .venv/bin/python scripts/run_framework_suite.py \
        --num-tasks "${ALF_NUM_TASKS}" \
        --splits valid_unseen \
        --frameworks "${FRAMEWORKS}" \
        --model "${MODEL}" \
        --seed "${seed}" \
        --runs-root "${ALF_ROOT}" \
        --memory-bank "${ALF_KB}" \
        --max-learnings "${ALF_MAX_LEARNINGS}" \
        --min-valid-level "${ALF_MIN_VALID_LEVEL}" \
        --quiet \
        --resume
  done
fi

if phase_enabled sql; then
  echo "[run] phase 3: SQL"
  for seed in ${SEEDS}; do
    echo "[run] refreshing SQL Docker before SQL seed ${seed}"
    docker compose -f data/intercode_sql/docker/docker-compose.yml restart
    run_logged "sql_seed_${seed}" \
      .venv/bin/python scripts/run_intercode_sql_suite.py \
        --num-tasks "${SQL_NUM_TASKS}" \
        --splits "${SQL_SPLITS}" \
        --frameworks "${FRAMEWORKS}" \
        --model "${MODEL}" \
        --seed "${seed}" \
        --runs-root "${SQL_ROOT}" \
        --memory-bank "${SQL_KB_SANITIZED}" \
        --max-learnings "${SQL_MAX_LEARNINGS}" \
        --min-valid-level "${SQL_MIN_VALID_LEVEL}" \
        --quiet \
        --resume
  done
fi

echo "[run] auditing GPT-5.4 matrix"
run_logged "audit" \
  .venv/bin/python scripts/cloud/audit_gpt54_matrix.py \
    --model "${MODEL}" \
    --web-root "${WEB_ROOT}" \
    --alf-root "${ALF_ROOT}" \
    --sql-root "${SQL_ROOT}" \
    --phases "${PHASES}" \
    --frameworks "${FRAMEWORKS}" \
    --seeds "${SEEDS}" \
    --web-expected "${WEBSHOP_NUM_TASKS}" \
    --alf-expected "${ALF_NUM_TASKS}" \
    --sql-expected "${SQL_NUM_TASKS}" \
    --web-split "${WEBSHOP_SPLITS}" \
    --sql-split "${SQL_SPLITS}"

echo "[run] complete"
